import argparse
from pathlib import Path
import numpy as np
import torch

from neurocircuit.models.model import NeurocircuitMechDecomp

def compute_lag_metrics(pi: np.ndarray):
    """Compute edge strength and lag statistics from π.

    pi: (T, R_tgt, R_src, L)
    Returns table rows (src, tgt, strength, peak_lag, centroid, concentration)
    """
    # Average over time for a stable summary
    p = pi.mean(axis=0)  # (R_tgt,R_src,L)
    Rt, Rs, L = p.shape
    lags = np.arange(L, dtype=np.float32)

    rows = []
    for j in range(Rt):
        for i in range(Rs):
            dist = p[j, i]
            strength = float(dist.sum())
            if strength <= 0:
                continue
            dist_norm = dist / (dist.sum() + 1e-12)
            peak = int(dist_norm.argmax())
            centroid = float((lags * dist_norm).sum())
            var = float(((lags - centroid) ** 2 * dist_norm).sum())
            concentration = float(1.0 / (var + 1e-6))
            rows.append((i, j, strength, peak, centroid, concentration))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--anat-mask', required=True, help='(R,R) boolean numpy mask of allowed edges')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--max-lag', type=int, default=13)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    y = data['Y'].astype(np.float32)  # (R,T)
    y_t = torch.from_numpy(y[None, ...])  # (B=1,R,T)

    anat_mask = np.load(args.anat_mask).astype(bool)
    anat_t = torch.from_numpy(anat_mask)

    # Build model (random init for prelim; plug in trained checkpoint when available)
    R = y.shape[0]
    model = NeurocircuitMechDecomp(n_regions=R, d_model=128, n_lags=args.max_lag)
    model.eval()

    with torch.no_grad():
        out = model(y_t, anat_mask=anat_t, max_lag=args.max_lag)

    pi = out['transformer']['pi'][0].cpu().numpy()  # (T,R_tgt,R_src,L)
    drive = out['transformer']['drive'][0].cpu().numpy()  # (R,T)

    np.save(outdir / 'routing_pi.npy', pi)
    np.save(outdir / 'drive_u.npy', drive)

    rows = compute_lag_metrics(pi)
    header = 'src,tgt,strength,peak_lag,centroid_lag,concentration\n'
    csv = [header] + [f"{i},{j},{s:.6g},{pk},{c:.6g},{conc:.6g}\n" for (i,j,s,pk,c,conc) in rows]
    (outdir / 'routing_metrics.csv').write_text(''.join(csv))

    print(f"Wrote: {outdir / 'routing_metrics.csv'}")


if __name__ == '__main__':
    main()
