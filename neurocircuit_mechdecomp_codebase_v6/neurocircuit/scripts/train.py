import argparse
import torch
from torch.optim import Adam
from neurocircuit.utils.config import load_yaml
from neurocircuit.utils.seed import seed_all
from neurocircuit.models.model import NeurocircuitMechDecomp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_all(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = cfg["model"]["n_regions"]
    model = NeurocircuitMechDecomp(
        n_regions=R,
        d_model=cfg["model"].get("d_model", 128),
        n_lags=cfg["model"].get("n_lags", 13),
    ).to(device)

    # TODO: replace with real data loader
    B, T = cfg["training"].get("batch_size", 2), cfg["data"].get("n_timepoints", 200)
    y = torch.randn(B, R, T, device=device)
    anat_mask = torch.ones(R, R, dtype=torch.bool, device=device)

    opt = Adam(model.parameters(), lr=cfg["training"].get("lr", 1e-4))

    for step in range(cfg["training"].get("steps", 10)):
        opt.zero_grad()
        out = model(y, anat_mask=anat_mask, max_lag=cfg["model"].get("n_lags", 13))

        # Strategy 1 stop-grad: do not backprop into SSM from transformer auxiliary loss
        x_hat = out["ssm"]["x_hat"].detach()

        # Placeholder loss (replace with NLL + forecasting, etc.)
        drive = out["transformer"]["drive"]
        loss = (drive ** 2).mean()

        loss.backward()
        opt.step()

        if step % 1 == 0:
            print(f"step {step:04d} | loss={loss.item():.6f}")

if __name__ == "__main__":
    main()
