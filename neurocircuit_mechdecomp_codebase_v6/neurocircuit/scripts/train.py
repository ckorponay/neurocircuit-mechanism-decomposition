from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from neurocircuit.models.model import NeurocircuitMechDecomp
from neurocircuit.training.losses import latent_dynamics_loss, routing_entropy_penalty
from neurocircuit.utils.config import load_yaml
from neurocircuit.utils.seed import seed_all


def build_model(cfg: dict) -> NeurocircuitMechDecomp:
    m = cfg["model"]
    attention_mode = m.get("attention_mode", "legacy_additive")
    return NeurocircuitMechDecomp(
        n_regions=int(m["n_regions"]),
        d_model=int(m.get("d_model", 128)),
        n_lags=m.get("n_lags", 13),
        lag_embedding_mode=m.get("lag_embedding_mode", "index"),
        max_lag_seconds=float(m.get("max_lag_seconds", 12.0)),
        attention_mode=attention_mode,
        lag_interaction_dim=int(m.get("lag_interaction_dim", 16)),
        legacy_oldest_first=bool(m.get("legacy_oldest_first", attention_mode == "legacy_additive")),
        ssm_parameterization=m.get("ssm_parameterization", "discrete"),
        ssm_initial_state=m.get("ssm_initial_state", "first_observation"),
        continuous_stability_mode=m.get("continuous_stability_mode", "projected"),
        discretization_method=m.get("discretization_method", "solve"),
        n_cortical_regions=int(m.get("n_cortical_regions", 0)),
        cortical_low_rank_rank=int(m.get("cortical_low_rank_rank", 0)),
    )


def _load_anat_mask(path: str | None, n_regions: int, device: torch.device) -> torch.Tensor:
    if path is None:
        return torch.ones(n_regions, n_regions, dtype=torch.bool, device=device)
    arr = np.load(path)
    if arr.shape != (n_regions, n_regions):
        raise ValueError(f"anatomical mask must be [{n_regions},{n_regions}], got {arr.shape}")
    return torch.as_tensor(arr.astype(bool), device=device)


def _load_latent_batch(path: str, device: torch.device) -> torch.Tensor:
    z = np.load(path)
    if "x_hat" not in z:
        raise ValueError("latent NPZ must contain x_hat [N,R,T] or [R,T]")
    x = np.asarray(z["x_hat"], dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"x_hat must be [N,R,T], got {x.shape}")
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train the circuit-dynamics M-step on fixed latent neural states. "
            "For production use, x_hat should first be inferred from BOLD with the "
            "rsHRF/RAPIDTIDE-aware alternating MAP inference path."
        )
    )
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--latent-npz",
        default=None,
        help="NPZ containing x_hat [N,R,T]. If omitted, runs a synthetic smoke test only.",
    )
    ap.add_argument("--anat-mask", default=None, help="Routing mask .npy [R_src,R_tgt] for Transformer drive")
    ap.add_argument("--dynamics-mask", default=None, help="Optional broader .npy [R_src,R_tgt] mask for SSM A; defaults to --anat-mask")
    ap.add_argument("--checkpoint-out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_all(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    R = int(cfg["model"]["n_regions"])
    tr_seconds = float(cfg["data"]["tr_seconds"])
    max_lag_seconds = float(cfg["model"].get("max_lag_seconds", 12.0))
    anat_mask = _load_anat_mask(args.anat_mask, R, device)
    dynamics_mask = (
        anat_mask if args.dynamics_mask is None
        else _load_anat_mask(args.dynamics_mask, R, device)
    )

    if args.latent_npz is None:
        B = int(cfg.get("training", {}).get("batch_size", 2))
        T = int(cfg["data"].get("n_timepoints", 200))
        x_hat = torch.randn(B, R, T, device=device)
        print(
            "[SMOKE ONLY] --latent-npz was not supplied. Training on synthetic latent "
            "states to verify gradients/coupling; do not interpret the fitted parameters."
        )
    else:
        x_hat = _load_latent_batch(args.latent_npz, device)
        if x_hat.shape[1] != R:
            raise ValueError(f"config n_regions={R}, but x_hat has R={x_hat.shape[1]}")

    train_cfg = cfg.get("training", {})
    opt = Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)))
    n_steps = int(train_cfg.get("steps", 25))
    entropy_weight = float(train_cfg.get("routing_entropy_weight", 0.0))

    for step in range(n_steps):
        opt.zero_grad()
        dyn_loss, details = latent_dynamics_loss(
            model,
            x_hat,
            anat_mask,
            tr_seconds=tr_seconds,
            max_lag_seconds=max_lag_seconds,
            dynamics_mask=dynamics_mask,
            return_details=True,
        )
        loss = dyn_loss
        ent = torch.zeros((), device=device)
        edge_mass = details["route"].get("edge_mass")
        if entropy_weight and edge_mass is not None:
            ent = routing_entropy_penalty(edge_mass)
            loss = loss + entropy_weight * ent

        loss.backward()
        opt.step()

        if step == 0 or (step + 1) % max(1, n_steps // 5) == 0:
            print(
                f"step {step + 1:04d}/{n_steps} | total={loss.item():.6g} | "
                f"dynamics={dyn_loss.item():.6g} | entropy={ent.item():.6g}"
            )

    if args.checkpoint_out:
        out = Path(args.checkpoint_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "note": "M-step checkpoint; verify E-step/inference provenance for production use.",
            },
            out,
        )
        print(f"wrote checkpoint: {out}")


if __name__ == "__main__":
    main()
