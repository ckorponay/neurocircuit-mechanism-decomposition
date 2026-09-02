from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def exact_zoh_discretize(
    A_c: torch.Tensor,
    B_c: torch.Tensor,
    dt: float,
    *,
    method: str = "solve",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-order-hold discretization of dx/dt = A_c x + B_c u.

    `solve` computes A_d = exp(A_c dt) and B_d = A_c^{-1}(A_d-I)B_c,
    avoiding a matrix exponential twice the state dimension. `augmented` uses
    the exact block-matrix exponential and is retained as a robust reference.
    """
    if dt <= 0:
        raise ValueError("tr_seconds must be > 0")
    if method not in {"solve", "augmented"}:
        raise ValueError("method must be 'solve' or 'augmented'")

    n = A_c.shape[0]
    if method == "solve":
        A_d = torch.matrix_exp(A_c * float(dt))
        eye = torch.eye(n, device=A_c.device, dtype=A_c.dtype)
        rhs = (A_d - eye) @ B_c
        try:
            B_d = torch.linalg.solve(A_c, rhs)
            return A_d, B_d
        except RuntimeError:
            # Rare ill-conditioning fallback.
            method = "augmented"

    m = B_c.shape[1]
    aug = torch.zeros((n + m, n + m), device=A_c.device, dtype=A_c.dtype)
    aug[:n, :n] = A_c
    aug[:n, n:] = B_c
    exp_aug = torch.matrix_exp(aug * float(dt))
    return exp_aug[:n, :n], exp_aug[:n, n:]


class LinearSSM(nn.Module):
    """
    Linear state-space dynamics with legacy and portable parameterizations.

    Recommended portable mode:
      parameterization='continuous'
      continuous_stability_mode='diagonal_dominant'

    The diagonal-dominant construction guarantees continuous-time stability by
    design and avoids an eigendecomposition/projection at every optimizer step.
    It can also enforce the anatomical graph directly on A_c.
    """

    def __init__(
        self,
        n_regions: int,
        rank_B: int = 3,
        parameterization: str = "discrete",
        initial_state: str = "first_observation",
        continuous_stability_margin: float = 1e-3,
        continuous_stability_mode: str = "projected",
        discretization_method: str = "solve",
        n_cortical_regions: int = 0,
        cortical_low_rank_rank: int = 0,
    ):
        super().__init__()
        if parameterization not in {"discrete", "continuous"}:
            raise ValueError("parameterization must be 'discrete' or 'continuous'")
        if initial_state not in {"first_observation", "zero"}:
            raise ValueError("initial_state must be 'first_observation' or 'zero'")
        if continuous_stability_mode not in {"projected", "diagonal_dominant"}:
            raise ValueError("unknown continuous_stability_mode")

        self.n = int(n_regions)
        self.parameterization = parameterization
        self.initial_state = initial_state
        self.continuous_stability_margin = float(continuous_stability_margin)
        self.continuous_stability_mode = continuous_stability_mode
        self.discretization_method = discretization_method
        self.n_cortical_regions = int(n_cortical_regions)
        self.cortical_low_rank_rank = int(cortical_low_rank_rank)
        if self.n_cortical_regions < 0 or self.n_cortical_regions > self.n:
            raise ValueError("n_cortical_regions must be in [0,n_regions]")
        if self.cortical_low_rank_rank < 0:
            raise ValueError("cortical_low_rank_rank must be >= 0")
        if self.cortical_low_rank_rank and (
            parameterization != "continuous" or continuous_stability_mode != "diagonal_dominant"
        ):
            raise ValueError(
                "cortical low-rank dynamics are supported in the recommended "
                "continuous + diagonal_dominant SSM mode"
            )

        if parameterization == "discrete":
            self.A = nn.Parameter(0.01 * torch.randn(n_regions, n_regions))
            self.register_parameter("A_c", None)
            self.register_parameter("A_offdiag", None)
            self.register_parameter("decay_raw", None)
        elif continuous_stability_mode == "projected":
            A0 = 0.01 * torch.randn(n_regions, n_regions)
            A0 = A0 - 0.20 * torch.eye(n_regions)
            self.A_c = nn.Parameter(A0)
            self.register_parameter("A", None)
            self.register_parameter("A_offdiag", None)
            self.register_parameter("decay_raw", None)
        else:
            # Off-diagonal couplings plus guaranteed negative diagonal.
            self.A_offdiag = nn.Parameter(0.01 * torch.randn(n_regions, n_regions))
            self.decay_raw = nn.Parameter(torch.full((n_regions,), -1.5))
            self.register_parameter("A", None)
            self.register_parameter("A_c", None)

        if self.cortical_low_rank_rank > 0:
            nc = self.n_cortical_regions
            rc = self.cortical_low_rank_rank
            self.cortical_U = nn.Parameter(0.005 * torch.randn(nc, rc))
            self.cortical_V = nn.Parameter(0.005 * torch.randn(nc, rc))
        else:
            self.register_parameter("cortical_U", None)
            self.register_parameter("cortical_V", None)

        self.U = nn.Parameter(0.01 * torch.randn(n_regions, rank_B))
        self.V = nn.Parameter(0.01 * torch.randn(n_regions, rank_B))

    def _mask_with_diagonal(self, dynamics_mask: torch.Tensor | None, device) -> torch.Tensor | None:
        if dynamics_mask is None:
            return None
        if dynamics_mask.shape != (self.n, self.n):
            raise ValueError("dynamics_mask must be [R_src,R_tgt]")
        # A rows are target, columns source, hence transpose graph convention.
        m = dynamics_mask.to(device=device, dtype=torch.bool).T.clone()
        idx = torch.arange(self.n, device=device)
        m[idx, idx] = True
        return m

    def effective_A_continuous(self, dynamics_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.parameterization != "continuous":
            raise ValueError("only defined for continuous parameterization")

        if self.continuous_stability_mode == "projected":
            A = self.A_c
            m = self._mask_with_diagonal(dynamics_mask, A.device)
            if m is not None:
                A = A * m.to(A.dtype)
            # Differentiable stability shift; no in-place eig projection.
            max_real = torch.linalg.eigvals(A).real.max()
            shift = torch.clamp(
                max_real + self.continuous_stability_margin,
                min=0.0,
            )
            eye = torch.eye(self.n, device=A.device, dtype=A.dtype)
            return A - shift * eye

        raw = self.A_offdiag
        eye_bool = torch.eye(self.n, device=raw.device, dtype=torch.bool)
        off = raw.masked_fill(eye_bool, 0.0)
        m = self._mask_with_diagonal(dynamics_mask, raw.device)
        if m is not None:
            off = off * (m & ~eye_bool).to(off.dtype)

        # Distributed cortex->cortex background dynamics are represented by a
        # low-rank block rather than 9,900 independently estimated edges. This
        # term is deliberately separate from the sparse anatomical mask and
        # from Transformer routing.
        if self.cortical_low_rank_rank > 0:
            nc = self.n_cortical_regions
            cblock = self.cortical_U @ self.cortical_V.T
            ceye = torch.eye(nc, device=raw.device, dtype=torch.bool)
            cblock = cblock.masked_fill(ceye, 0.0)
            off = off.clone()
            off[:nc, :nc] = off[:nc, :nc] + cblock

        # Strict row diagonal dominance with negative diagonal guarantees all
        # Gershgorin discs lie in the open left half-plane, including the
        # low-rank cortical contribution.
        row_abs = off.abs().sum(dim=1)
        extra_decay = F.softplus(self.decay_raw) + self.continuous_stability_margin
        diag = -(row_abs + extra_decay)
        return off + torch.diag(diag)


    def cortical_low_rank_matrix(self) -> torch.Tensor | None:
        """Return the directed cortical background A block before diagonal stabilization."""
        if self.cortical_low_rank_rank <= 0:
            return None
        block = self.cortical_U @ self.cortical_V.T
        eye = torch.eye(
            self.n_cortical_regions, device=block.device, dtype=torch.bool
        )
        return block.masked_fill(eye, 0.0)

    def stability_project_(self, rho_max: float = 0.98):
        """Legacy discrete projection only; portable stable-by-design mode needs none."""
        if self.parameterization != "discrete":
            return
        with torch.no_grad():
            rho = torch.linalg.eigvals(self.A).abs().max().real
            if rho > rho_max:
                self.A.mul_(rho_max / (rho + 1e-8))

    def continuous_matrices(
        self,
        dynamics_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.parameterization != "continuous":
            raise ValueError("continuous_matrices requires continuous parameterization")
        A_c = self.effective_A_continuous(dynamics_mask=dynamics_mask)
        B_c = self.U @ self.V.T
        return A_c, B_c

    def step_matrices(
        self,
        tr_seconds: float | None = None,
        dynamics_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.parameterization == "discrete":
            self.stability_project_()
            A = self.A
            m = self._mask_with_diagonal(dynamics_mask, A.device)
            if m is not None:
                A = A * m.to(A.dtype)
            return A, self.U @ self.V.T

        if tr_seconds is None:
            raise ValueError("continuous SSM requires tr_seconds")
        A_c, B_c = self.continuous_matrices(dynamics_mask=dynamics_mask)
        return exact_zoh_discretize(
            A_c,
            B_c,
            float(tr_seconds),
            method=self.discretization_method,
        )

    def _initial_x(self, y: torch.Tensor) -> torch.Tensor:
        if self.initial_state == "zero":
            return torch.zeros_like(y[:, :, 0])
        return y[:, :, 0]

    def transition_prediction(
        self,
        x: torch.Tensor,
        *,
        u: torch.Tensor | None,
        tr_seconds: float | None,
        dynamics_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict x[:, :, 1:] from x[:, :, :-1] and optional drive u."""
        A_d, B_d = self.step_matrices(tr_seconds, dynamics_mask=dynamics_mask)
        pred = x[:, :, :-1].transpose(1, 2) @ A_d.T
        if u is not None:
            if u.shape != x.shape:
                raise ValueError("u must match x shape [B,R,T]")
            pred = pred + (u[:, :, :-1].transpose(1, 2) @ B_d.T)
        return pred.transpose(1, 2)

    def forward(
        self,
        y: torch.Tensor,
        u: torch.Tensor | None = None,
        tr_seconds: float | None = None,
        dynamics_mask: torch.Tensor | None = None,
    ) -> dict:
        Bsz, R, T = y.shape
        A_step, B_step = self.step_matrices(tr_seconds, dynamics_mask=dynamics_mask)

        x = torch.zeros((Bsz, R, T), device=y.device, dtype=y.dtype)
        x[:, :, 0] = self._initial_x(y)
        for t in range(T - 1):
            inp = (
                torch.zeros((Bsz, R), device=y.device, dtype=y.dtype)
                if u is None
                else u[:, :, t]
            )
            x[:, :, t + 1] = (x[:, :, t] @ A_step.T) + (inp @ B_step.T)

        out = {"x_hat": x, "A_d": A_step, "B_d": B_step}
        if self.parameterization == "discrete":
            out.update(
                {
                    "A": A_step,
                    "B": self.U @ self.V.T,
                    "legacy_spectral_radius": torch.linalg.eigvals(A_step).abs().max().real,
                }
            )
            return out

        A_c, B_c = self.continuous_matrices(dynamics_mask=dynamics_mask)
        eig_c = torch.linalg.eigvals(A_c)
        max_real = eig_c.real.max()
        decay = torch.where(
            max_real < 0,
            -1.0 / max_real,
            torch.full_like(max_real, float("inf")),
        )
        out.update(
            {
                "A": A_c,
                "B": B_c,
                "A_c": A_c,
                "B_c": B_c,
                "stability_margin_per_second": -max_real,
                "slowest_decay_seconds": decay,
                "tr_seconds": float(tr_seconds),
            }
        )
        return out
