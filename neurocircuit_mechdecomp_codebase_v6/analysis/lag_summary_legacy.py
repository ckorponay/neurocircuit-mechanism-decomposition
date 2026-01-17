"""
Lag Summary — Which lags best predict Striatum after training?
--------------------------------------------------------------
This utility scans attention logs from a trained loop‑aware model and
summarizes, for each cross‑edge (e.g., C→S, Th→S, A→S, H→S), **how much
attention mass is placed on each time lag** in the context window.

What it tells you
-----------------
• A distribution over lags (in TRs and seconds) per edge → interpret as
  "which past moments from this source are most informative for S now?"
• Scalar summaries: **mean lag**, **median lag**, **peak lag**, and central intervals.
• Per‑subject CSV + cohort‑level aggregates; optional plots.

Assumptions (matches our model/data)
------------------------------------
• For each cross‑edge key in logs['cross'], attention has shape (B,H,N_tgt,N_src)
  where **N_src = T_ctx * P_src** (all source tokens across all lags), flattened
  in time‑major order. We reshape to (T_ctx, P_src) to recover lags.
• The attention series corresponds to the **last frame** of each window, so the
  lag mapping is: `lag_tr = (T_ctx - 1) - tpos` (tpos ∈ [0..T_ctx-1]).
• Attention is *credit assignment*, not causality; we report it as such.

Outputs
-------
• `per_subject_lag_summary.csv` — one row per subject×run×edge with:
   subject, task, run, edge, mean_lag_tr, mean_lag_s, median_lag_s, peak_lag_s,
   p25_lag_s, p75_lag_s
• `edge_<SRC>_to_<TGT>_lag_hist.csv` — cohort‑aggregated mass per lag (columns:
   lag_tr, lag_s, mass)
• Optional PDF `lag_plots.pdf` with per‑edge histograms (toggle `--plot`).

Example
-------
python lag_summary.py \
  --cfg configs/rest_pretrain_normative.json \
  --ckpt outputs/ckpts/loop_v2_burstlog_gate_best.pt \
  --list data/npz_rest/rest_list.txt \
  --tr 0.72 --outdir outputs/lag_rest --edges C->S Th->S A->S H->S --plot
"""
from __future__ import annotations
import os, re, json, argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Import model API ---
try:
    import loopaware_v2 as loop
except Exception:
    import loop_loopaware_cst_transformer as loop

BIDS_PAT = re.compile(r"sub-([A-Za-z0-9]+)(?:[_/].*?ses-([A-Za-z0-9]+))?.*?task-([A-Za-z0-9]+).*?(?:run-([A-Za-z0-9]+))?", re.IGNORECASE)

def bids_fields(path: str):
    m = BIDS_PAT.search(path)
    if not m:
        base = os.path.basename(path).replace('.npz','')
        return {'sub': base, 'ses': None, 'task': None, 'run': None}
    sub, ses, task, run = m.groups()
    return {'sub': sub, 'ses': ses, 'task': (task or '').lower(), 'run': run}


def reshape_src(A: torch.Tensor, T_ctx: int, P_src: int) -> torch.Tensor:
    """A: (B,H,N_tgt,N_src) → (B,H,N_tgt,T_ctx,P_src) with checks."""
    B,H,Nt,Ns = A.shape
    assert Ns % P_src == 0, f"N_src={Ns} not divisible by P_src={P_src}"
    assert Ns // P_src == T_ctx, f"Expected T_ctx={T_ctx} from cfg, got {Ns//P_src}"
    return A.view(B,H,Nt,T_ctx,P_src)


def collect_lag_mass_for_edge(model, ds, device, edge: Tuple[str,str], T_ctx: int, P_src: int) -> np.ndarray:
    """Traverse dataset; accumulate attention mass per lag for a given edge.
    Returns a vector length T_ctx with mass per lag (TR units)."""
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    lag_mass = np.zeros(T_ctx, dtype=np.float64)
    with torch.no_grad():
        for batch in loader:
            X = {k: v.to(device) for k,v in batch['X'].items()}
            last_masked = {k: v.to(device) for k,v in batch['last_frame_masked'].items()}
            _, _, logs = model(X, last_masked)
            key = f"{edge[0]}->{edge[1]}"
            if 'cross' not in logs or key not in logs['cross']:
                continue
            A = logs['cross'][key]  # (B,H,N_tgt,N_src)
            A = reshape_src(A, T_ctx, P_src)        # (B,H,N_tgt,T_ctx,P_src)
            # Sum over heads, targets, and src tokens → mass per tpos
            # Then map tpos → lag: lag_tr = (T_ctx - 1) - tpos
            mass_tpos = A.sum(dim=(0,1,2,4)).detach().cpu().numpy()  # (T_ctx,)
            # Reverse to get by-lag smallest first: 1 TR ago near index 1, etc.
            mass_lag = mass_tpos[::-1]
            lag_mass += mass_lag
    return lag_mass


def summarize_from_hist(lag_mass: np.ndarray, tr: float):
    idx = np.arange(len(lag_mass))  # 0..T_ctx-1 (0 = same frame; 1 = 1 TR ago, ...)
    m = lag_mass.astype(np.float64)
    tot = m.sum()
    if tot <= 0:
        return {k: np.nan for k in ['mean_tr','mean_s','median_s','peak_s','p25_s','p75_s']}
    pmf = m / tot
    mean_tr = float((idx * pmf).sum())
    cdf = np.cumsum(pmf)
    def q(p):
        return float(idx[np.searchsorted(cdf, p, side='left')])
    median = q(0.5); p25 = q(0.25); p75 = q(0.75)
    peak = float(idx[np.argmax(pmf)])
    return {
        'mean_tr': mean_tr,
        'mean_s': mean_tr * tr,
        'median_s': median * tr,
        'peak_s': peak * tr,
        'p25_s': p25 * tr,
        'p75_s': p75 * tr,
    }


def main():
    ap = argparse.ArgumentParser(description='Summarize average lags per edge using attention logs')
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--list', required=True, help='Text file with NPZ paths (one per line)')
    ap.add_argument('--tr', type=float, required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--edges', nargs='*', default=None, help='Edges like C->S Th->S A->S H->S')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.cfg,'r') as f:
        cfgd = json.load(f)
    cfg = loop.ModelConfig(**cfgd); cfg.device = device
    state = torch.load(args.ckpt, map_location=device)
    model = loop.LoopAwareModel(cfg).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()

    # Build list
    with open(args.list,'r') as f:
        paths = [ln.strip() for ln in f if ln.strip()]

    # Determine edges & P_src lookup from cfg
    have_edges = set(tuple(e) for e in cfg.edges)
    if args.edges:
        E = [tuple(e.split('->')) for e in args.edges]
    else:
        E = [("C","S"),("Th","S"),("A","S"),("H","S")]
    edges = [e for e in E if e in have_edges]

    # P_src by group
    P_map = {"C": cfg.P_C, "Th": cfg.P_Th, "A": cfg.P_A, "H": cfg.P_H, "S": cfg.P_S,
             "GPe": cfg.P_GPe, "GPi": cfg.P_GPi, "STN": cfg.P_STN, "MB": cfg.P_MB}

    # Cohort accumulators
    cohort_hist: Dict[str, np.ndarray] = {f"{s}->{t}": np.zeros(cfg.T_ctx, dtype=np.float64) for (s,t) in edges}
    rows = []  # per‑subject summary rows

    for npz_path in tqdm(paths, desc='subjects'):
        # Subject/run fields
        bf = bids_fields(npz_path)
        sid = bf['sub']; task = bf['task'] or 'na'; run = bf['run'] or 'na'
        # Build dataset for this single run
        tmp_list = os.path.join(args.outdir, '_tmp_one.txt')
        with open(tmp_list,'w') as f:
            f.write(npz_path)
        ds = loop.LoopDataset(tmp_list, cfg, mask_ratio=0.4)
        for (s,t) in edges:
            key = f"{s}->{t}"
            P_src = P_map[s]
            lag_mass = collect_lag_mass_for_edge(model, ds, device, (s,t), cfg.T_ctx, P_src)
            cohort_hist[key] += lag_mass
            summ = summarize_from_hist(lag_mass, args.tr)
            rows.append({
                'subject': sid, 'task': task, 'run': run, 'edge': key,
                'mean_lag_tr': summ['mean_tr'], 'mean_lag_s': summ['mean_s'],
                'median_lag_s': summ['median_s'], 'peak_lag_s': summ['peak_s'],
                'p25_lag_s': summ['p25_s'], 'p75_lag_s': summ['p75_s']
            })

    # Write per‑subject summary
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, 'per_subject_lag_summary.csv'), index=False)

    # Write cohort histograms per edge
    for key, mass in cohort_hist.items():
        lag_tr = np.arange(cfg.T_ctx)
        lag_s  = lag_tr * args.tr
        out = pd.DataFrame({'lag_tr': lag_tr, 'lag_s': lag_s, 'mass': mass})
        out.to_csv(os.path.join(args.outdir, f'edge_{key.replace("->","_to_")}_lag_hist.csv'), index=False)

    # Optional plots
    if args.plot:
        pdf = os.path.join(args.outdir, 'lag_plots.pdf')
        with PdfPages(pdf) as pp:
            for key, mass in cohort_hist.items():
                plt.figure(figsize=(8,4))
                x = np.arange(cfg.T_ctx) * args.tr
                y = mass / (mass.sum() + 1e-12)
                plt.plot(x, y)
                plt.xlabel('Lag (s)'); plt.ylabel('Attention mass (fraction)')
                plt.title(f'Lag distribution: {key}')
                pp.savefig(); plt.close()
        print(f"[OK] wrote {pdf}")

    print(f"[OK] wrote per_subject_lag_summary.csv and edge_*_lag_hist.csv in {args.outdir}")

if __name__ == '__main__':
    main()
