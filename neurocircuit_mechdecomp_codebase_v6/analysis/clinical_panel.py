"""
Clinical Circuit Panel — Subject vs Normative
--------------------------------------------
Generates a compact, multi-panel PDF summarizing a subject's cortico‑striatal metrics
from our attention model, with **normative comparisons** and flags. Works with the
outputs from `burst_state_analysis_gate.py` and (optionally) its `states`/`hazard` steps.

Inputs
------
Required:
  --series   subject series.csv (from `burst_state_analysis_gate.py eval`)
Optional but recommended:
  --states   subject states.csv (from `... states`)
  --hazard   subject hazard_report.json (from `... hazard`)
Normative reference (one of):
  --norm-metrics CSV with columns: metric, mean, std, p05, p50, p95
  OR
  --norm-samples CSV with columns: metric, value  (we compute mean/std/percentiles)

Demo mode (if you just want to SEE the layout):
  --demo  (ignores missing normative; fabricates a reasonable reference from subject)

Output
------
  --out PDF path (e.g., outputs/panel_sub-XXXX.pdf)

Metrics visualized
------------------
Summary tiles:
  • Up occupancy (% of frames in Up or Burst‑Up)
  • Burst rate (per minute; TD_total & MSI_total high‑coincidence frames)
  • Mean TD in Up and in Burst‑Up
  • Hazard 50% threshold TD (optional if hazard provided)
Edge-level bars:
  • Mean TD by excitatory edge to S (C→S, Th→S, A→S, H→S)
Gating:
  • MB gate scalars by edge (distributions + subject mean)
Distributions:
  • TD_total histogram vs normative band; MSI_total histogram
Flags table:
  • Any metric with |z| ≥ 2 highlighted
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

# ----------------------
# Helpers
# ----------------------

def load_normative(norm_metrics_csv: str | None, norm_samples_csv: str | None, metrics_needed: list[str], demo_fallback: dict[str, float] | None = None):
    stats = {}
    if norm_metrics_csv and os.path.exists(norm_metrics_csv):
        nm = pd.read_csv(norm_metrics_csv)
        for m in metrics_needed:
            row = nm.loc[nm['metric']==m]
            if len(row):
                r = row.iloc[0]
                stats[m] = {'mean': float(r.get('mean', np.nan)), 'std': float(r.get('std', np.nan)), 'p05': float(r.get('p05', np.nan)), 'p50': float(r.get('p50', np.nan)), 'p95': float(r.get('p95', np.nan))}
    elif norm_samples_csv and os.path.exists(norm_samples_csv):
        ns = pd.read_csv(norm_samples_csv)
        for m in metrics_needed:
            x = ns.loc[ns['metric']==m, 'value'].values
            if x.size:
                stats[m] = {'mean': float(np.mean(x)), 'std': float(np.std(x, ddof=1)+1e-8), 'p05': float(np.quantile(x,0.05)), 'p50': float(np.quantile(x,0.50)), 'p95': float(np.quantile(x,0.95))}
    # demo fallback
    if demo_fallback:
        for m in metrics_needed:
            if m not in stats:
                mu = float(demo_fallback.get(f'{m}_mu', 0.0))
                sd = float(abs(demo_fallback.get(f'{m}_sd', 1.0)) + 1e-6)
                stats[m] = {'mean': mu, 'std': sd, 'p05': mu - 1.64*sd, 'p50': mu, 'p95': mu + 1.64*sd}
    return stats


def zscore(x: float, mu: float, sd: float) -> float:
    if sd <= 0 or np.isnan(sd): return np.nan
    return (x - mu) / sd


def compute_subject_metrics(series_csv: str, states_csv: str | None) -> dict:
    df = pd.read_csv(series_csv)
    metrics = {}
    # Up occupancy (Up or Burst-Up = state > 0)
    if states_csv and os.path.exists(states_csv):
        st = pd.read_csv(states_csv)
        m = df.merge(st[['t_index','state']], on='t_index', how='left')
        up_occ = float((m['state'] > 0).mean())
        metrics['up_occupancy'] = up_occ
        # TD means by state
        if m['state'].notna().any():
            for label, cond in [('td_mean_up', m['state']==1), ('td_mean_burst', m['state']==2 if 'p_state2' in st.columns or (m['state'].max()>=2) else (m['TD_total']>=np.quantile(m['TD_total'],0.98)) )]:
                sel = cond
                if isinstance(sel, pd.Series):
                    val = float(m.loc[sel, 'TD_total'].mean()) if sel.any() else np.nan
                else:
                    val = float(m.loc[sel, 'TD_total'].mean()) if np.any(sel) else np.nan
                metrics[label] = val
    else:
        metrics['up_occupancy'] = np.nan
        metrics['td_mean_up'] = np.nan
        metrics['td_mean_burst'] = np.nan

    # Burst rate per minute (subject-defined bursts: TD>q98 & MSI>q75)
    qTD, qMSI = np.quantile(df['TD_total'].values, 0.98), np.quantile(df['MSI_total'].values, 0.75)
    bursts = (df['TD_total'].values > qTD) & (df['MSI_total'].values > qMSI)
    # Approx seconds per point: unknown TR; infer from t_sec diff if available
    if 't_sec' in df.columns and df['t_sec'].notna().sum() > 1:
        sec_span = float(df['t_sec'].dropna().iloc[-1] - df['t_sec'].dropna().iloc[0])
        minutes = max(sec_span/60.0, 1e-6)
    else:
        minutes = max(len(df)/60.0, 1e-6)  # assume 1 Hz if TR unknown
    metrics['burst_rate_per_min'] = float(bursts.sum() / minutes)

    # Edge TD means (excitatory only if available)
    for edge in ['C->S','Th->S','A->S','H->S']:
        col = f'TD:{edge}'
        if col in df.columns:
            metrics[f'mean_TD_{edge}'] = float(df[col].mean())
    # Gate means
    for edge in ['C->S','Th->S','A->S','H->S']:
        col = f'Gate:{edge}'
        if col in df.columns:
            metrics[f'mean_Gate_{edge}'] = float(pd.to_numeric(df[col], errors='coerce').mean())

    # Distribution summaries for TD_total / MSI_total
    metrics['TD_total_mean'] = float(df['TD_total'].mean())
    metrics['TD_total_p95'] = float(np.quantile(df['TD_total'], 0.95))
    metrics['MSI_total_mean'] = float(df['MSI_total'].mean())

    return metrics, df


def load_hazard(hazard_json: str | None) -> dict:
    if hazard_json and os.path.exists(hazard_json):
        with open(hazard_json,'r') as f:
            rep = json.load(f)
        return rep
    return {}


# ----------------------
# Panel drawing
# ----------------------

def draw_panel(subject_metrics: dict, df_series: pd.DataFrame, norm_stats: dict, hazard: dict, out_pdf: str, title: str = 'Clinical Circuit Panel'):
    # Metrics to norm
    need = ['up_occupancy','burst_rate_per_min','td_mean_up','td_mean_burst','TD_total_mean','MSI_total_mean']
    for e in ['C->S','Th->S','A->S','H->S']:
        need.append(f'mean_TD_{e}')
        need.append(f'mean_Gate_{e}')
    # Build z-scores
    z = {}
    for m in need:
        if m in subject_metrics and m in norm_stats and not np.isnan(subject_metrics[m]):
            z[m] = zscore(subject_metrics[m], norm_stats[m]['mean'], norm_stats[m]['std'])
        else:
            z[m] = np.nan

    # Layout
    fig = plt.figure(figsize=(12,9))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.1,1.1,1.0], hspace=0.6, wspace=0.4)

    # A: Summary tiles (row 0, col 0:3)
    axA = fig.add_subplot(gs[0, :])
    tiles = [
        ('Up occupancy', 'up_occupancy', '%', 100*subject_metrics.get('up_occupancy', np.nan), 100*norm_stats.get('up_occupancy',{}).get('p50', np.nan), 100*norm_stats.get('up_occupancy',{}).get('p05', np.nan), 100*norm_stats.get('up_occupancy',{}).get('p95', np.nan)),
        ('Burst rate (/min)', 'burst_rate_per_min', '', subject_metrics.get('burst_rate_per_min', np.nan), norm_stats.get('burst_rate_per_min',{}).get('p50', np.nan), norm_stats.get('burst_rate_per_min',{}).get('p05', np.nan), norm_stats.get('burst_rate_per_min',{}).get('p95', np.nan)),
        ('TD mean (Up)', 'td_mean_up', '', subject_metrics.get('td_mean_up', np.nan), norm_stats.get('td_mean_up',{}).get('p50', np.nan), norm_stats.get('td_mean_up',{}).get('p05', np.nan), norm_stats.get('td_mean_up',{}).get('p95', np.nan)),
        ('TD mean (Burst-Up)', 'td_mean_burst', '', subject_metrics.get('td_mean_burst', np.nan), norm_stats.get('td_mean_burst',{}).get('p50', np.nan), norm_stats.get('td_mean_burst',{}).get('p05', np.nan), norm_stats.get('td_mean_burst',{}).get('p95', np.nan)),
    ]
    x = np.arange(len(tiles))
    for i,(lab,key,unit,val,p50,p05,p95) in enumerate(tiles):
        axA.plot([i-0.35,i+0.35],[p50,p50], lw=2)
        axA.fill_between([i-0.35,i+0.35],[p05,p05],[p95,p95], alpha=0.2)
        axA.scatter([i],[val], marker='o')
        axA.text(i, val, f"  {val:.2f}{unit}", va='bottom', fontsize=9)
    axA.set_xticks(x); axA.set_xticklabels([t[0] for t in tiles], rotation=0)
    axA.set_title('Summary (point = subject, line = norm median, band = 5–95%)')

    # B: TD_total distribution vs norm band (row1 col0)
    axB = fig.add_subplot(gs[1,0])
    td = df_series['TD_total'].dropna().values
    axB.hist(td, bins=40, alpha=0.7)
    if 'TD_total_mean' in norm_stats:
        mu = norm_stats['TD_total_mean']['mean']; sd = norm_stats['TD_total_mean']['std']
        axB.axvline(mu, ls='--'); axB.axvspan(mu-2*sd, mu+2*sd, alpha=0.15)
    axB.set_title('TD_total distribution (subject)')
    axB.set_xlabel('TD_total'); axB.set_ylabel('count')

    # C: Edge TD bars (row1 col1)
    axC = fig.add_subplot(gs[1,1])
    edges = ['C->S','Th->S','A->S','H->S']
    vals = [subject_metrics.get(f'mean_TD_{e}', np.nan) for e in edges]
    axC.bar(np.arange(len(edges)), vals)
    axC.set_xticks(np.arange(len(edges))); axC.set_xticklabels(edges)
    axC.set_title('Mean TD by excitatory edge → S')

    # D: Gate means (row1 col2)
    axD = fig.add_subplot(gs[1,2])
    gvals = [subject_metrics.get(f'mean_Gate_{e}', np.nan) for e in edges]
    axD.bar(np.arange(len(edges)), gvals)
    axD.set_xticks(np.arange(len(edges))); axD.set_xticklabels(edges)
    axD.set_title('MB gate (mean scalar) by edge')

    # E: MSI_total distribution (row2 col0)
    axE = fig.add_subplot(gs[2,0])
    msi = df_series['MSI_total'].dropna().values
    axE.hist(msi, bins=40, alpha=0.7)
    if 'MSI_total_mean' in norm_stats:
        mu = norm_stats['MSI_total_mean']['mean']; sd = norm_stats['MSI_total_mean']['std']
        axE.axvline(mu, ls='--'); axE.axvspan(mu-2*sd, mu+2*sd, alpha=0.15)
    axE.set_title('MSI_total distribution (subject)'); axE.set_xlabel('MSI_total'); axE.set_ylabel('count')

    # F: Flags table (row2 col1:2)
    axF = fig.add_subplot(gs[2,1:])
    rows = []
    for m in ['up_occupancy','burst_rate_per_min','td_mean_up','td_mean_burst'] + [f'mean_TD_{e}' for e in edges] + [f'mean_Gate_{e}' for e in edges]:
        if m in z and not np.isnan(z[m]):
            if abs(z[m]) >= 2.0:
                rows.append((m, subject_metrics.get(m, np.nan), z[m]))
    txt = 'Abnormal (|z| ≥ 2):\n' + ('\n'.join([f"• {k}: value={v:.3f}, z={zs:.2f}" for k,v,zs in rows]) if rows else '• none')
    axF.axis('off'); axF.text(0,1, txt, va='top', family='monospace')

    fig.suptitle(title, fontsize=14)
    with PdfPages(out_pdf) as p:
        p.savefig(fig)
    plt.close(fig)
    print(f"[OK] wrote {out_pdf}")


# ----------------------
# CLI
# ----------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Generate clinical panel PDF for subject vs normative reference')
    ap.add_argument('--series', required=True)
    ap.add_argument('--states', default=None)
    ap.add_argument('--hazard', default=None)
    ap.add_argument('--norm-metrics', default=None)
    ap.add_argument('--norm-samples', default=None)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    subj_metrics, df_series = compute_subject_metrics(args.series, args.states)
    haz = load_hazard(args.hazard)

    # Prepare demo fallback based on subject to make something sensible
    fallback = None
    if args.demo:
        fallback = {
            'up_occupancy_mu': max(min(subj_metrics.get('up_occupancy', 0.3), 0.8), 0.1), 'up_occupancy_sd': 0.1,
            'burst_rate_per_min_mu': max(subj_metrics.get('burst_rate_per_min', 1.0)*0.8, 0.2), 'burst_rate_per_min_sd': max(subj_metrics.get('burst_rate_per_min', 1.0)*0.4, 0.2),
            'td_mean_up_mu': df_series['TD_total'].mean()*0.9, 'td_mean_up_sd': df_series['TD_total'].std()*0.6 + 1e-6,
            'td_mean_burst_mu': df_series['TD_total'].quantile(0.95), 'td_mean_burst_sd': df_series['TD_total'].std()*0.8 + 1e-6,
            'TD_total_mean_mu': df_series['TD_total'].mean()*0.9, 'TD_total_mean_sd': df_series['TD_total'].std()*0.5 + 1e-6,
            'MSI_total_mean_mu': df_series['MSI_total'].mean()*0.95, 'MSI_total_mean_sd': df_series['MSI_total'].std()*0.5 + 1e-6,
        }
        for e in ['C->S','Th->S','A->S','H->S']:
            fallback[f'mean_TD_{e}_mu'] = subj_metrics.get(f'mean_TD_{e}', df_series['TD_total'].mean()/4)
            fallback[f'mean_TD_{e}_sd'] = abs(df_series['TD_total'].std()/6) + 1e-6
            fallback[f'mean_Gate_{e}_mu'] = 1.1
            fallback[f'mean_Gate_{e}_sd'] = 0.15
    metrics_needed = ['up_occupancy','burst_rate_per_min','td_mean_up','td_mean_burst','TD_total_mean','MSI_total_mean'] + [f'mean_TD_{e}' for e in ['C->S','Th->S','A->S','H->S']] + [f'mean_Gate_{e}' for e in ['C->S','Th->S','A->S','H->S']]

    norm = load_normative(args.norm_metrics, args.norm_samples, metrics_needed, demo_fallback=fallback)

    draw_panel(subj_metrics, df_series, norm, haz, args.out, title='Clinical Circuit Panel — Subject vs Normative')
