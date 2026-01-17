"""
Burst & State Analysis (with MB gate logging)
--------------------------------------------
Computes **co-drive burst metrics** (TD/MSI) from attention logs + value norms,
logs **MB gate scalars** where available (Gate:<edge>), optionally fits a 2–3 state
HMM (Down / Up / Burst-Up), and creates CSV + PDF outputs.

Compatible with `loopaware_v2_burstlog_gate.py`.

Install deps (once):
    pip install numpy pandas matplotlib tqdm hmmlearn scikit-learn

Usage
-----
# 1) Per-subject burst metrics (+ optional peri-event overlay) from a trained model
python burst_state_analysis_gate.py eval \
  --cfg configs/gambling_eval.json \
  --ckpt outputs/ckpts/loop_v2_burstlog_gate_best.pt \
  --npz  data/sub-100307_tfMRI_GAMBLING_LR.npz \
  --tr 0.72 --outdir outputs/burst_sub-100307 \
  --edges C->S A->S H->S Th->S GPe->S \
  --events data/sub-100307_task-gambling_events.tsv  # optional

# 2) Fit HMM states on the saved CSV (2- or 3-state) and plot
python burst_state_analysis_gate.py states \
  --csv outputs/burst_sub-100307/series.csv \
  --states 3 --outdir outputs/burst_sub-100307

# 3) Hazard analysis of Down→Up transitions
python burst_state_analysis_gate.py hazard \
  --series outputs/burst_sub-100307/series.csv \
  --states outputs/burst_sub-100307/states.csv \
  --outdir outputs/burst_sub-100307 \
  --gpe-key TD:GPe->S   # optional; leave unset to ignore

What it computes
----------------
Per frame (aligned to the *last* frame of each context window):
- For each edge (e.g., C->S):
   • contributions c_i(t) = mean_head_target(alpha) * ||value|| per source i
   • TD_edge(t) = sum_i c_i(t)
   • MSI_edge(t) = fraction of sources above their own 90th percentile (computed over time)
   • Gate_edge(t) = MB gate scalar (if gating active and MB present)
- Pooled across chosen edges:
   • TD_total(t) = sum_edges TD_edge(t)
   • MSI_total(t) computed on concatenated sources across edges
- Also saves mean S amplitude at that frame (mean over S tokens)
"""
from __future__ import annotations
import os, json, argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import the burst+gate-logging model
import loopaware_v2_burstlog_gate as loop

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def mean_over_heads_targets(A: torch.Tensor) -> torch.Tensor:
    """A: (B,H,N_tgt,N_src) -> (B,N_src) by averaging over H and N_tgt."""
    return A.mean(dim=1).mean(dim=2)

# -----------------------------
# EVAL: collect TD/MSI(+Gate) series
# -----------------------------

def eval_subject(cfg_path: str, ckpt_path: str, npz_path: str, tr: float, outdir: str,
                 edges: Optional[List[Tuple[str,str]]] = None,
                 events_tsv: Optional[str] = None,
                 msi_quantile: float = 0.9):
    ensure_dir(outdir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(cfg_path,'r') as f: cfgd = json.load(f)
    cfg = loop.ModelConfig(**cfgd); cfg.device = device
    state = torch.load(ckpt_path, map_location=device)
    model = loop.LoopAwareModel(cfg).to(device)
    model.load_state_dict(state['state_dict']); model.eval()

    tmp_list = os.path.join(outdir, '_tmp_list.txt')
    with open(tmp_list,'w') as f: f.write(npz_path)
    ds = loop.LoopDataset(tmp_list, cfg, mask_ratio=0.4)
    ld = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    have_edges = set(tuple(e) for e in cfg.edges)
    if edges is None:
        candidates = [("C","S"),("A","S"),("H","S"),("Th","S"),("GPe","S")]
    else:
        candidates = edges
    edges_used = [e for e in candidates if e in have_edges]

    contrib_series = {}   # edge -> list of (1,N_src) numpy
    TD_series = {}        # edge -> list of floats
    Gate_series = {}      # edge -> list of floats (or NaN)
    Xs_mean = []          # mean S amplitude at last frame

    with torch.no_grad():
        for batch in tqdm(ld, desc='collect', leave=False):
            X = {k: v.to(device) for k,v in batch['X'].items()}
            last_masked = {k: v.to(device) for k,v in batch['last_frame_masked'].items()}
            if 'S' in batch['X']:
                Xs_mean.append(float(batch['X']['S'][:, -1].mean().item()))
            _, _, logs = model(X, last_masked)
            A_cross: Dict[str, torch.Tensor] = logs['cross']
            V_cross: Dict[str, torch.Tensor] = logs.get('cross_vnorm', {})
            G_cross: Dict[str, Optional[torch.Tensor]] = logs.get('cross_gate', {})
            for (src,tgt) in edges_used:
                key = f"{src}->{tgt}"
                if key not in A_cross or key not in V_cross or V_cross[key] is None:
                    continue
                A = A_cross[key]      # (B,H,N_tgt,N_src)
                Vn = V_cross[key]     # (B,N_src)
                alpha = mean_over_heads_targets(A).squeeze(0)       # (N_src)
                vnorm = Vn.squeeze(0)                                # (N_src)
                c = (alpha * vnorm).detach().cpu().numpy()[None, :] # (1,N_src)
                contrib_series.setdefault(key, []).append(c)
                TD_series.setdefault(key, []).append(float(c.sum()))
                gval = np.nan
                if key in G_cross and G_cross[key] is not None:
                    try:
                        gval = float(G_cross[key].squeeze(0).detach().cpu().item())
                    except Exception:
                        gval = np.nan
                Gate_series.setdefault(key, []).append(gval)

    # Stack over time
    time = np.arange(len(next(iter(TD_series.values()))) ) if TD_series else np.arange(len(Xs_mean))
    per_edge = {}
    TD_total = np.zeros_like(time, dtype=float)
    MSI_total = np.zeros_like(time, dtype=float)

    # Compute per-edge MSI using per-source quantiles
    for key, lst in contrib_series.items():
        C = np.vstack(lst)[:,0,:]  # T x N_src
        q = np.quantile(C, msi_quantile, axis=0)
        above = (C > q[None,:]).astype(float)
        MSI = above.mean(axis=1)  # fraction of sources above their own high quantile
        TD = np.array(TD_series[key])
        per_edge[key] = {'TD': TD, 'MSI': MSI, 'N_src': C.shape[1]}
        TD_total += TD
        if 'MSI_pool' not in locals():
            MSI_pool = above
        else:
            MSI_pool = np.concatenate([MSI_pool, above], axis=1)
    if 'MSI_pool' in locals():
        MSI_total = MSI_pool.mean(axis=1)

    # Save time series CSV
    rows = []
    for t in range(len(time)):
        row = {'t_index': t, 't_sec': float('nan'), 'TD_total': TD_total[t], 'MSI_total': MSI_total[t], 'S_mean': Xs_mean[t] if t < len(Xs_mean) else float('nan')}
        for key, d in per_edge.items():
            row[f'TD:{key}'] = d['TD'][t]; row[f'MSI:{key}'] = d['MSI'][t]
        for key, lst in Gate_series.items():
            if t < len(lst): row[f'Gate:{key}'] = lst[t]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df)>0 and tr is not None:
        df['t_sec'] = (cfg.T_ctx + cfg.k_forecast - 1 + df['t_index']) * tr
    out_csv = os.path.join(outdir, 'series.csv'); df.to_csv(out_csv, index=False)

    # Basic plots
    pdf = os.path.join(outdir, 'series_plots.pdf')
    with PdfPages(pdf) as p:
        plt.figure(figsize=(9,4)); plt.plot(df['t_index'], df['TD_total']); plt.title('Total Co-drive (TD)'); plt.xlabel('Window index'); plt.ylabel('TD'); p.savefig(); plt.close()
        plt.figure(figsize=(9,4)); plt.plot(df['t_index'], df['MSI_total']); plt.title('Multi-Source Index (MSI)'); plt.xlabel('Window index'); plt.ylabel('MSI'); p.savefig(); plt.close()
        # Gates
        gate_cols = [c for c in df.columns if c.startswith('Gate:')]
        for gc in gate_cols:
            plt.figure(figsize=(9,3)); plt.plot(df['t_index'], df[gc]); plt.title(gc.replace('Gate:','Gate — ')); plt.xlabel('Window index'); plt.ylabel('MB gate (scalar)'); p.savefig(); plt.close()
    print(f"[OK] wrote {out_csv} and {pdf}")

    # Optional: peri-event overlay if events provided
    if events_tsv and os.path.exists(events_tsv):
        try:
            ev = pd.read_csv(events_tsv, sep='\t')
            if not {'onset','duration'}.issubset(ev.columns):
                raise ValueError('events TSV missing onset/duration')
            onsets = (ev['onset'].values / tr).round().astype(int) - (cfg.T_ctx + cfg.k_forecast - 1)
            pre, post = 8, 16
            X = np.arange(-pre, post+1)
            def peri(series):
                mats = []
                for o in onsets:
                    a=o-pre; b=o+post
                    if a<0 or b>=len(series): continue
                    mats.append(series[a:b+1])
                return np.vstack(mats).mean(axis=0) if mats else np.zeros(len(X))
            pdf2 = os.path.join(outdir, 'peri_event_td_msi.pdf')
            with PdfPages(pdf2) as p:
                plt.figure(figsize=(9,4)); plt.plot(X*tr, peri(df['TD_total'].values)); plt.axvline(0, ls='--'); plt.title('Peri-event TD_total'); plt.xlabel('s'); plt.ylabel('TD'); p.savefig(); plt.close()
                plt.figure(figsize=(9,4)); plt.plot(X*tr, peri(df['MSI_total'].values)); plt.axvline(0, ls='--'); plt.title('Peri-event MSI_total'); plt.xlabel('s'); plt.ylabel('MSI'); p.savefig(); plt.close()
            print(f"[OK] wrote {pdf2}")
        except Exception as e:
            print(f"[WARN] Peri-event overlay skipped: {e}")

# -----------------------------
# STATES: HMM on derived features
# -----------------------------

def fit_states(csv_path: str, outdir: str, n_states: int = 3, stickiness: float = 0.95):
    ensure_dir(outdir)
    df = pd.read_csv(csv_path)
    feats = df[['TD_total','MSI_total','S_mean']].copy()
    X = feats.values.astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        raise RuntimeError("Install hmmlearn: pip install hmmlearn")
    hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, tol=1e-3)
    startprob = np.zeros(n_states); startprob[0] = 1.0
    transmat = np.full((n_states,n_states), (1.0 - stickiness)/(n_states-1))
    np.fill_diagonal(transmat, stickiness)
    hmm.startprob_ = startprob
    hmm.transmat_ = transmat
    hmm.fit(X)
    z = hmm.predict(X)
    post = hmm.predict_proba(X)
    df_states = pd.DataFrame({'t_index': df['t_index'], 'state': z})
    for k in range(n_states): df_states[f'p_state{k}'] = post[:,k]
    out_states = os.path.join(outdir, 'states.csv'); df_states.to_csv(out_states, index=False)
    pdf = os.path.join(outdir, 'states_plots.pdf')
    with PdfPages(pdf) as p:
        plt.figure(figsize=(10,3)); plt.plot(df['t_index'], df['TD_total'], label='TD'); plt.plot(df['t_index'], df['MSI_total'], label='MSI'); plt.legend(); plt.title('TD & MSI'); p.savefig(); plt.close()
        plt.figure(figsize=(10,3)); plt.plot(df_states['t_index'], df_states['p_state0'], label='p(Down)');
        if n_states>1: plt.plot(df_states['t_index'], df_states['p_state1'], label='p(Up)');
        if n_states>2: plt.plot(df_states['t_index'], df_states['p_state2'], label='p(Burst-Up)');
        plt.ylim(0,1); plt.legend(); plt.title('State posteriors'); p.savefig(); plt.close()
    print(f"[OK] wrote {out_states} and {pdf}")

# -----------------------------
# Hazard: Down→Up transition modeling
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def hazard(series_csv: str, states_csv: str, outdir: str, include_gpe: bool = True, gpe_key: str = 'TD:GPe->S'):
    ensure_dir(outdir)
    df = pd.read_csv(series_csv)
    st = pd.read_csv(states_csv)
    M = df.merge(st[['t_index','state']], on='t_index', how='inner').sort_values('t_index')
    rows = []
    for t in range(len(M)-1):
        s = int(M.iloc[t]['state']); s_next = int(M.iloc[t+1]['state'])
        if s == 0:
            y = 1 if s_next > 0 else 0
            feats = {'TD_total': float(M.iloc[t]['TD_total']), 'MSI_total': float(M.iloc[t]['MSI_total'])}
            if include_gpe and gpe_key in M.columns:
                feats['TD_GPeS'] = float(M.iloc[t][gpe_key])
            rows.append({**feats, 'y': y})
    if not rows:
        print('[WARN] No Down frames to model transitions.'); return
    D = pd.DataFrame(rows).fillna(0.0)
    X_base = D[['TD_total','MSI_total']].values; y = D['y'].values
    lr_base = LogisticRegression(max_iter=300).fit(X_base, y)
    auc_base = roc_auc_score(y, lr_base.predict_proba(X_base)[:,1])
    auc_full, delta_auc, coef = auc_base, 0.0, {'intercept': float(lr_base.intercept_[0]), 'TD_total': float(lr_base.coef_[0,0]), 'MSI_total': float(lr_base.coef_[0,1])}
    if include_gpe and 'TD_GPeS' in D.columns:
        X_full = D[['TD_total','MSI_total','TD_GPeS']].values
        lr_full = LogisticRegression(max_iter=300).fit(X_full, y)
        auc_full = roc_auc_score(y, lr_full.predict_proba(X_full)[:,1])
        delta_auc = float(auc_full - auc_base)
        coef.update({'TD_GPeS': float(lr_full.coef_[0,2])})
    rep = {'n_samples': int(len(D)), 'auc_base_TD_MSI': float(auc_base), 'auc_with_GPe': float(auc_full), 'delta_auc_GPe': float(delta_auc), 'coefficients': coef}
    with open(os.path.join(outdir,'hazard_report.json'),'w') as f: json.dump(rep, f, indent=2)
    td = np.linspace(D['TD_total'].quantile(0.01), D['TD_total'].quantile(0.99), 100)
    msi_med = np.median(D['MSI_total'])
    if include_gpe and 'TD_GPeS' in D.columns:
        gpe_med = np.median(D['TD_GPeS'])
        proba = lr_full.predict_proba(np.c_[td, np.full_like(td, msi_med), np.full_like(td, gpe_med)])[:,1]
    else:
        proba = lr_base.predict_proba(np.c_[td, np.full_like(td, msi_med)])[:,1]
    pdf = os.path.join(outdir,'hazard_plots.pdf')
    with PdfPages(pdf) as p:
        plt.figure(figsize=(8,4)); plt.plot(td, proba); plt.xlabel('TD_total'); plt.ylabel('P(Up at t+1 | Down at t)'); plt.title('Hazard vs TD_total'); p.savefig(); plt.close()
    print(f"[OK] wrote hazard_report.json and hazard_plots.pdf in {outdir}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Burst (TD/MSI) + gates, states, and hazard analysis for loop-aware model')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('eval', help='Compute TD/MSI (+Gate) time series (and optional peri-event overlay)')
    p1.add_argument('--cfg', required=True)
    p1.add_argument('--ckpt', required=True)
    p1.add_argument('--npz', required=True)
    p1.add_argument('--tr', type=float, required=True)
    p1.add_argument('--outdir', required=True)
    p1.add_argument('--edges', nargs='*', default=None, help='Edges e.g. C->S A->S H->S Th->S GPe->S')
    p1.add_argument('--events', default=None, help='Optional BIDS events.tsv for peri-event overlays')

    p2 = sub.add_parser('states', help='Fit HMM states on a saved series.csv')
    p2.add_argument('--csv', required=True)
    p2.add_argument('--states', type=int, default=3)
    p2.add_argument('--outdir', required=True)

    p3 = sub.add_parser('hazard', help='Model Down→Up transitions as a function of TD/MSI (+/- GPe)')
    p3.add_argument('--series', required=True)
    p3.add_argument('--states', required=True, dest='states_csv')
    p3.add_argument('--outdir', required=True)
    p3.add_argument('--gpe-key', default='TD:GPe->S')

    args = ap.parse_args()
    if args.cmd == 'eval':
        edges = [tuple(e.split('->')) for e in args.edges] if args.edges else None
        eval_subject(args.cfg, args.ckpt, args.npz, args.tr, args.outdir, edges, args.events)
    elif args.cmd == 'states':
        fit_states(args.csv, args.outdir, n_states=args.states)
    elif args.cmd == 'hazard':
        hazard(args.series, args.states_csv, args.outdir, include_gpe=(args.gpe_key is not None and len(args.gpe_key)>0), gpe_key=args.gpe_key)
