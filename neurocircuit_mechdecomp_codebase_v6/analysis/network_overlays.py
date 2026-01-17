"""
Network overlays (post‑hoc) — TR‑wise large‑scale cortical networks
-------------------------------------------------------------------
Adds **Phase‑1** overlays without changing the model: compute TR‑wise
**cortical network (CN)** time series from your existing NPZs and relate them
to your learned striatal metrics (TD, MSI, gate, states, hazards).

What this script does
---------------------
1) **Compute CN time series** from the cortex tokens (C) in an NPZ by pooling
   tokens that belong to the same large‑scale network (e.g., Yeo‑7, Yeo‑17).
   Methods: **mean** (default) or **first PC** per network.
2) **Summarize & correlate** with your series metrics (if provided):
   • Pearson r of each CN with **TD_total**, **MSI_total**, and **MB_gate**
   • Mean CN level within **Down/Up/Burst‑Up** states (if states.csv given)
3) **Optional hazard augmentation** (if you pass both series.csv and states.csv):
   Compare a baseline hazard model P(Up@t+1 | Down@t, TD, MSI, Gate) vs
   augmented with CNs. Reports ΔAUC and coefficients.
4) **Optional plots**: per‑network traces and peri‑event overlay around burst onsets.

Inputs
------
• `--npz`            : one NPZ with a 'C' array of shape (T, P_C).
• Network mapping (choose one of):
  (A) `--token2net`  : TSV with columns `token` (0‑based cortex token index) and `network` (name).
  (B) `--c-index` + `--label2net` : use `C_index.tsv` from converter (token→label) and a
      TSV mapping `label`→`network`.
• Optional: `--series` your per‑TR metrics CSV (from burst_state_analysis_gate.py)
            expected columns: `t`, `TD_total`, `MSI_total`, `MB_gate`, `state` (0=Down,1=Up,2=Burst‑Up)
• Optional: `--states` if states are in a separate file (one int per TR).
• Optional: `--tr` repetition time (sec) for plots; defaults to 1 if omitted.

Outputs (all in `--outdir`)
---------------------------
• `cn_series.csv`       : columns `t`, `t_sec`, one column per network.
• `cn_summary.csv`      : r(CN, TD/MSI/gate) and mean CN per state.
• `hazard_compare.csv`  : AUCs and coefficients (if hazard run is requested).
• `cn_plots.pdf`        : optional figure with CN time series and peri‑event averages.

Examples
--------
# Minimal: compute CN series with a token→network TSV
python network_overlays.py \
  --npz data/npz_rest/sub-01_rest.npz \
  --token2net maps/schaefer400_token2yeo7.tsv \
  --tr 0.72 --outdir outputs/sub-01_CN

# Use converter outputs: token→label and label→network mapping TSV
python network_overlays.py \
  --npz data/npz_rest/sub-01_rest.npz \
  --c-index data/npz_rest/meta/C_index.tsv \
  --label2net maps/schaefer400_label2yeo7.tsv \
  --series outputs/sub-01_rest/series.csv --states outputs/sub-01_rest/states.csv \
  --tr 0.72 --plots --hazard

Notes
-----
• This is **post‑hoc only**: no model changes, no extra tokens.
• Network names are free‑text; the script preserves your names in outputs.
• If your series/states filenames differ, use `--state-col` and `--time-col` flags.
"""
from __future__ import annotations
import os, argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception:
    PCA = None
    LogisticRegression = None
    roc_auc_score = None

# ---------------------
# IO helpers
# ---------------------

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_token2net(path: str) -> Dict[int, str]:
    df = pd.read_csv(path, sep='\t' if path.endswith('.tsv') else ',', dtype={'token': int, 'network': str})
    if not {'token','network'}.issubset(df.columns):
        raise ValueError('token2net must have columns token, network')
    return {int(r.token): str(r.network) for r in df.itertuples(index=False)}


def load_cindex_and_label2net(c_index: str, label2net: str) -> Dict[int, str]:
    ci = pd.read_csv(c_index, sep='\t')
    if not {'token','label'}.issubset(ci.columns):
        raise ValueError('C_index.tsv must have columns token, label')
    l2n = pd.read_csv(label2net, sep='\t' if label2net.endswith('.tsv') else ',')
    if not {'label','network'}.issubset(l2n.columns):
        raise ValueError('label2net must have columns label, network')
    l2n_dict = {int(r.label): str(r.network) for r in l2n.itertuples(index=False)}
    out: Dict[int,str] = {}
    for r in ci.itertuples(index=False):
        lab = int(r.label)
        if lab in l2n_dict:
            out[int(r.token)] = l2n_dict[lab]
    return out

# ---------------------
# CN computation
# ---------------------

def compute_cn_series(C: np.ndarray, token2net: Dict[int,str], method: str = 'mean', pca_var: float = 0.9) -> Tuple[pd.DataFrame, List[str]]:
    """Return DataFrame with one column per network and rows=TR."""
    T, P = C.shape
    nets = sorted(set(token2net.values()))
    X = {}
    for net in nets:
        idx = [i for i in range(P) if token2net.get(i)==net]
        if len(idx)==0:
            X[net] = np.zeros(T, dtype=np.float32)
            continue
        Ci = C[:, idx]
        if method == 'mean':
            X[net] = Ci.mean(axis=1)
        elif method in ('pc1','pca'):
            if PCA is None:
                raise RuntimeError('Install scikit-learn for PCA method')
            pca = PCA(n_components=1)
            X[net] = pca.fit_transform(Ci).reshape(-1)
        else:
            raise ValueError("method must be 'mean' or 'pc1'")
    df = pd.DataFrame(X)
    return df, nets

# ---------------------
# Summaries & correlations
# ---------------------

def summarize_cn(df_cn: pd.DataFrame, series_csv: Optional[str], states_csv: Optional[str], tr: float, outdir: str):
    rows = []
    if series_csv is not None and os.path.exists(series_csv):
        s = pd.read_csv(series_csv)
        # permissive column names
        td_col = next((c for c in s.columns if c.lower().startswith('td')), None)
        msi_col = next((c for c in s.columns if c.lower().startswith('msi')), None)
        gate_col = next((c for c in s.columns if 'gate' in c.lower()), None)
        # align length
        n = min(len(s), len(df_cn))
        s = s.iloc[:n]
        cn = df_cn.iloc[:n]
        for net in cn.columns:
            r_td = cn[net].corr(s[td_col]) if td_col else np.nan
            r_msi = cn[net].corr(s[msi_col]) if msi_col else np.nan
            r_gate = cn[net].corr(s[gate_col]) if gate_col else np.nan
            rows.append({'network': net, 'r_TD': r_td, 'r_MSI': r_msi, 'r_gate': r_gate})
    # Means per state
    if states_csv is not None and os.path.exists(states_csv):
        st = pd.read_csv(states_csv)
        if st.shape[1]==1:
            st.columns = ['state']
        if 'state' not in st.columns:
            # try first column
            st = st.rename(columns={st.columns[0]:'state'})
        n = min(len(st), len(df_cn))
        st = st.iloc[:n]
        cn = df_cn.iloc[:n]
        for net in cn.columns:
            for k,v in {0:'Down',1:'Up',2:'Burst-Up'}.items():
                m = float(cn.loc[st['state']==k, net].mean()) if (st['state']==k).any() else np.nan
                rows.append({'network': net, 'state_mean': v, 'value': m})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, 'cn_summary.csv'), index=False)

# ---------------------
# Hazard augmentation (optional)
# ---------------------

def hazard_with_cn(series_csv: str, states_csv: str, df_cn: pd.DataFrame, outdir: str):
    if LogisticRegression is None or roc_auc_score is None:
        print('[WARN] scikit-learn not installed; skipping hazard analysis')
        return
    s = pd.read_csv(series_csv)
    st = pd.read_csv(states_csv)
    if st.shape[1]==1:
        st.columns=['state']
    n = min(len(s), len(st), len(df_cn))
    s = s.iloc[:n]; st = st.iloc[:n]; cn = df_cn.iloc[:n]
    # Build labels: y = 1 if state[t]==0 (Down) and state[t+1]==1 (Up); else 0. Mask last TR.
    y = np.zeros(n-1, dtype=int)
    mask = (st['state'].values[:-1]==0)
    y[mask] = (st['state'].values[1:][mask]==1).astype(int)
    # Features baseline: TD, MSI, gate at t where Down
    td_col = next((c for c in s.columns if c.lower().startswith('td')), None)
    msi_col = next((c for c in s.columns if c.lower().startswith('msi')), None)
    gate_col = next((c for c in s.columns if 'gate' in c.lower()), None)
    X_base = np.stack([s[td_col].values[:-1], s[msi_col].values[:-1], s[gate_col].values[:-1]], axis=1)
    X_base = X_base[mask]
    y_base = y[mask]
    # With CNs: add all networks as regressors (z‑score columns)
    X_cn = np.hstack([X_base, cn.values[:-1][mask]])
    X_base = (X_base - X_base.mean(0)) / (X_base.std(0)+1e-6)
    X_cn = (X_cn - X_cn.mean(0)) / (X_cn.std(0)+1e-6)
    # Fit L2‑regularized logistic models
    lr_base = LogisticRegression(max_iter=200, solver='liblinear')
    lr_cn   = LogisticRegression(max_iter=200, solver='liblinear')
    lr_base.fit(X_base, y_base)
    lr_cn.fit(X_cn, y_base)
    # AUCs via in‑sample (for quick comparison); you can replace with CV if desired
    auc_base = roc_auc_score(y_base, lr_base.predict_proba(X_base)[:,1])
    auc_cn   = roc_auc_score(y_base, lr_cn.predict_proba(X_cn)[:,1])
    # Coefficients (map back to names)
    cols_base = ['TD','MSI','Gate']
    cols_cn = cols_base + [f'CN_{c}' for c in df_cn.columns]
    coef_base = pd.DataFrame({'feature': cols_base, 'coef': lr_base.coef_.reshape(-1)})
    coef_cn   = pd.DataFrame({'feature': cols_cn,   'coef': lr_cn.coef_.reshape(-1)})
    out = {
        'auc_base': auc_base,
        'auc_with_cn': auc_cn,
        'delta_auc': auc_cn - auc_base
    }
    pd.DataFrame([out]).to_csv(os.path.join(outdir,'hazard_compare.csv'), index=False)
    coef_base.to_csv(os.path.join(outdir,'hazard_coef_base.csv'), index=False)
    coef_cn.to_csv(os.path.join(outdir,'hazard_coef_with_cn.csv'), index=False)

# ---------------------
# Plotting (optional)
# ---------------------

def make_plots(df_cn: pd.DataFrame, series_csv: Optional[str], tr: float, outdir: str):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        print('[WARN] matplotlib not available; skipping plots')
        return
    pdf = os.path.join(outdir, 'cn_plots.pdf')
    with PdfPages(pdf) as pp:
        # CN traces
        plt.figure(figsize=(10,4))
        t = np.arange(len(df_cn))*tr
        for col in df_cn.columns:
            plt.plot(t, df_cn[col], label=col)
        plt.xlabel('Time (s)'); plt.ylabel('CN (a.u.)'); plt.title('Cortical network time series')
        plt.legend(ncol=4, fontsize=8)
        pp.savefig(); plt.close()
        # Peri‑burst overlay if series provided
        if series_csv is not None and os.path.exists(series_csv):
            s = pd.read_csv(series_csv)
            if 'burst_onset' in s.columns:
                onsets = np.where(s['burst_onset'].values.astype(int)==1)[0]
            else:
                # Heuristic: mark top 5% TD frames as bursts for visualization only
                td_col = next((c for c in s.columns if c.lower().startswith('td')), None)
                if td_col:
                    thr = np.nanpercentile(s[td_col].values, 95)
                    onsets = np.where(s[td_col].values >= thr)[0]
                else:
                    onsets = np.array([], dtype=int)
            pre, post = 8, 16
            X = np.arange(-pre, post+1)*tr
            for col in df_cn.columns:
                mats = []
                for o in onsets:
                    a = o - pre; b = o + post
                    if a < 0 or b >= len(df_cn):
                        continue
                    mats.append(df_cn[col].values[a:b+1])
                if len(mats)==0:
                    continue
                M = np.vstack(mats).mean(axis=0)
                plt.figure(figsize=(6,3))
                plt.plot(X, M)
                plt.axvline(0, ls='--', lw=1)
                plt.xlabel('Time from burst (s)'); plt.ylabel(col)
                plt.title(f'Peri‑burst CN — {col}')
                pp.savefig(); plt.close()
    print(f"[OK] wrote {pdf}")

# ---------------------
# CLI
# ---------------------

def main():
    ap = argparse.ArgumentParser(description='Post‑hoc cortical network overlays (TR‑wise)')
    ap.add_argument('--npz', required=True, help='NPZ file with array C (T, P_C)')
    # Network mapping options
    ap.add_argument('--token2net', help='TSV/CSV with columns token,network')
    ap.add_argument('--c-index', help='C_index.tsv from converter (token→label)')
    ap.add_argument('--label2net', help='TSV/CSV mapping label→network')
    # Method
    ap.add_argument('--method', choices=['mean','pc1'], default='mean')
    ap.add_argument('--tr', type=float, default=1.0)
    ap.add_argument('--outdir', required=True)
    # Optional series/states for summaries/hazard
    ap.add_argument('--series', default=None, help='series.csv with TD/MSI/gate (optional)')
    ap.add_argument('--states', default=None, help='states.csv with a column state (optional)')
    ap.add_argument('--hazard', action='store_true', help='Run hazard augmentation with CN covariates')
    ap.add_argument('--plots', action='store_true', help='Emit PDF plots')

    args = ap.parse_args()
    ensure_dir(args.outdir)

    npz = np.load(args.npz)
    if 'C' not in npz:
        raise ValueError('NPZ does not contain cortex array C')
    C = np.array(npz['C'])
    T, P = C.shape

    # Build token→network mapping
    if args.token2net:
        t2n = load_token2net(args.token2net)
    elif args.c_index and args.label2net:
        t2n = load_cindex_and_label2net(args.c_index, args.label2net)
    else:
        raise SystemExit('Provide either --token2net or both --c-index and --label2net')

    # Compute series
    df_cn, nets = compute_cn_series(C, t2n, method=args.method)
    # Save cn_series.csv
    df_out = df_cn.copy()
    df_out.insert(0, 't', np.arange(len(df_out)))
    df_out.insert(1, 't_sec', df_out['t']*args.tr)
    df_out.to_csv(os.path.join(args.outdir,'cn_series.csv'), index=False)

    # Summaries & correlations
    summarize_cn(df_cn, args.series, args.states, args.tr, args.outdir)

    # Hazard augmentation
    if args.hazard and args.series and args.states:
        hazard_with_cn(args.series, args.states, df_cn, args.outdir)

    # Plots
    if args.plots:
        make_plots(df_cn, args.series, args.tr, args.outdir)

    print('[OK] CN overlays complete')

if __name__ == '__main__':
    main()
