"""
Hazard / Transition Analysis for Striatal Up/Down vs Co‑drive (TD/MSI)
----------------------------------------------------------------------
Purpose: quantify how much **intrinsic state** (Down→Up threshold driven by co‑drive)
explains transitions, and how much extra predictive value an **external inhibitory
term** (e.g., GPe→S contribution) adds beyond that.

Inputs
------
- `series.csv` from `burst_state_analysis.py eval` (contains TD/MSI and per‑edge TDs)
- `states.csv` from `burst_state_analysis.py states` (contains discrete state + posteriors)

Outputs
-------
- `hazard_report.json` with:
  • sample counts, AUC for base model (TD+MSI), AUC with GPe term, ΔAUC
  • logistic coefficients (for interpretability)
- `hazard_plots.pdf` with partial‑dependence curve of P(Up at t+1 | Down at t)
  vs TD_total (MSI and GPe fixed at median).

Usage
-----
python hazard_transition_analysis.py \
  --series outputs/burst_sub-XXXX/series.csv \
  --states outputs/burst_sub-XXXX/states.csv \
  --outdir outputs/burst_sub-XXXX \
  --gpe-key "TD:GPe->S"   # optional; if absent, the analysis runs without GPe

Interpretation
--------------
- If ΔAUC with the GPe term is ~0 and its coefficient is tiny/NS, that supports the
  view that **hyperpolarizing control is predominantly intrinsic** at the TR scale.
- You can repeat with other putative inhibitory/modulatory edges by changing `--gpe-key`.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def run(series_csv: str, states_csv: str, outdir: str, gpe_key: str | None = "TD:GPe->S"):
    ensure_dir(outdir)
    df = pd.read_csv(series_csv)
    st = pd.read_csv(states_csv)

    # Merge on t_index; we assume state 0 was initialized as Down in the HSMM/HMM
    M = df.merge(st[['t_index','state']], on='t_index', how='inner').sort_values('t_index').reset_index(drop=True)
    rows = []
    for t in range(len(M)-1):
        s = int(M.loc[t,'state']); s_next = int(M.loc[t+1,'state'])
        if s == 0:  # Down at t
            y = 1 if s_next > 0 else 0  # Up or Burst-Up at t+1
            feats = {
                'TD_total': float(M.loc[t,'TD_total']),
                'MSI_total': float(M.loc[t,'MSI_total'])
            }
            if gpe_key and gpe_key in M.columns:
                feats['GPe_term'] = float(M.loc[t, gpe_key])
            rows.append({**feats, 'y': y})
    if not rows:
        raise RuntimeError('No Down frames to model transitions — check your states.csv.')

    D = pd.DataFrame(rows).fillna(0.0)
    X_base = D[['TD_total','MSI_total']].values
    y = D['y'].values

    lr_base = LogisticRegression(max_iter=500).fit(X_base, y)
    auc_base = roc_auc_score(y, lr_base.predict_proba(X_base)[:,1])

    coef = {
        'intercept_base': float(lr_base.intercept_[0]),
        'TD_total': float(lr_base.coef_[0,0]),
        'MSI_total': float(lr_base.coef_[0,1])
    }

    auc_full = auc_base
    delta_auc = 0.0
    if 'GPe_term' in D.columns:
        X_full = D[['TD_total','MSI_total','GPe_term']].values
        lr_full = LogisticRegression(max_iter=500).fit(X_full, y)
        auc_full = roc_auc_score(y, lr_full.predict_proba(X_full)[:,1])
        delta_auc = float(auc_full - auc_base)
        coef.update({'intercept_full': float(lr_full.intercept_[0]), 'GPe_term': float(lr_full.coef_[0,2])})

    rep = {
        'n_samples': int(len(D)),
        'auc_base_TD_MSI': float(auc_base),
        'auc_with_GPe': float(auc_full),
        'delta_auc_GPe': float(delta_auc),
        'coefficients': coef
    }
    with open(os.path.join(outdir,'hazard_report.json'),'w') as f:
        json.dump(rep, f, indent=2)

    # Partial dependence over TD_total (MSI & GPe fixed to medians)
    td = np.linspace(D['TD_total'].quantile(0.01), D['TD_total'].quantile(0.99), 120)
    msi_med = float(np.median(D['MSI_total']))
    if 'GPe_term' in D.columns:
        gpe_med = float(np.median(D['GPe_term']))
        proba = lr_full.predict_proba(np.c_[td, np.full_like(td, msi_med), np.full_like(td, gpe_med)])[:,1]
    else:
        proba = lr_base.predict_proba(np.c_[td, np.full_like(td, msi_med)])[:,1]

    pdf = os.path.join(outdir, 'hazard_plots.pdf')
    with PdfPages(pdf) as p:
        plt.figure(figsize=(8,4))
        plt.plot(td, proba)
        plt.xlabel('TD_total (co-drive)')
        plt.ylabel('P(Up at t+1 | Down at t)')
        plt.title('Hazard vs co-drive (MSI & GPe fixed at median)')
        p.savefig(); plt.close()

    print(f"[OK] wrote {os.path.join(outdir,'hazard_report.json')} and {pdf}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Down→Up hazard analysis vs co-drive (TD/MSI) with optional GPe term')
    ap.add_argument('--series', required=True)
    ap.add_argument('--states', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--gpe-key', default='TD:GPe->S', help='Column name in series.csv to treat as external inhibitory term; leave blank to ignore')
    args = ap.parse_args()
    gpe_key = None if (args.gpe_key is None or len(args.gpe_key.strip())==0) else args.gpe_key
    run(args.series, args.states, args.outdir, gpe_key)
