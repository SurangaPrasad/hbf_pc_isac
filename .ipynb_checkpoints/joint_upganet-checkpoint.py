"""Joint deep-unfolding PGA for sub-connected mmWave MIMO ISAC beamforming.

This module folds the antenna-to-RF-chain connectivity matrix ``S`` *into* the
same unfolded projected-gradient-ascent (PGA) iteration as the analog precoder
``F``, instead of learning ``S`` with a separate front-end network.  The whole
objective is

    g(F, W, S) = omega * R(F_eff, W) + log(CRLB(F_eff, W)^-1),

with ``F_eff = F * S`` (elementwise / Hadamard product), ``F`` the complex
analog precoder, ``S`` a real row-stochastic connection matrix, and ``W`` the
digital precoder.

Everything below operates on a leading batch dimension::

    F        : (B, N_antennas, N_rf)   complex
    S        : (B, N_antennas, N_rf)   real, row-stochastic (rows sum to 1)
    W        : (B, N_rf, N_users)      complex
    H        : (B, N_antennas, N_users) complex channel
    M_matrix : (B, N_antennas, N_antennas) Hermitian PSD sensing Fisher-like matrix
        (``A_dot^H R_N_inv A_dot``) in antenna space; also accepts a shared
        ``(N_antennas, N_antennas)`` matrix broadcast over the batch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Import the existing learnable antenna->RF assignment front-end.  It produces the
# *initial* connection matrix S_0; the unfolded layers then refine S (and F, W)
# jointly inside every outer iteration.
from SelectionNet import SelectionNet


# /////////////////////////////////////////////////////////////////////////////////////////
#                             TASK 1: ROW-WISE SIMPLEX PROJECTION
# /////////////////////////////////////////////////////////////////////////////////////////

def project_to_simplex_rows(X: torch.Tensor) -> torch.Tensor:
    """Project each row of ``X`` onto the probability simplex (Euclidean).

    Projects the last dimension of a real tensor onto the set
    ``{v : v_i >= 0, sum_i v_i = 1}`` using the O(d log d) sort-based algorithm
    of Duchi et al. (2008) / Held, Wolfe & Crowder (1974).  The algorithm sorts
    each row in descending order, finds the number of leading sorted entries
    whose value exceeds the running "water level", subtracts that water level
    ``theta``, and clips negatives to zero.  It is fully vectorized: arbitrary
    leading dimensions (e.g. ``(B, N, M)``) are supported and no Python loop
    runs over batch or row indices.

    Parameters
    ----------
    X : torch.Tensor
        Real tensor of shape ``(..., M)``; the last dimension is projected.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape with each last-dim slice a probability vector.
    """
    d = X.shape[-1]

    # 1. Sort each row in descending order: u_1 >= u_2 >= ... >= u_d.
    u, _ = torch.sort(X, dim=-1, descending=True)

    # 2. Running (cumulative) sums of the sorted values: c_j = sum_{i<=j} u_i.
    cumsum = torch.cumsum(u, dim=-1)

    # 3. Water levels per position: (c_j - 1) / j  (j is 1-indexed).
    j_idx = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    water_levels = (cumsum - 1.0) / j_idx

    # 4. rho = number of leading sorted entries satisfying u_j > water_level_j.
    #    Because u is sorted, the "> level" condition is monotone, so counting
    #    the True entries yields the largest such index directly.
    above_water = u > water_levels
    rho = above_water.long().sum(dim=-1)          # (...,)

    # 5. theta = (c_rho - 1) / rho: the amount subtracted from every entry.
    rho = torch.clamp(rho, min=1)
    rho_idx = (rho - 1).unsqueeze(-1)              # 0-indexed position in cumsum
    c_rho = torch.gather(cumsum, dim=-1, index=rho_idx).squeeze(-1)  # (...,)
    theta = (c_rho - 1.0) / rho.to(dtype=X.dtype)   # (...,)

    # 6. v = max(x - theta, 0).  The trailing unsqueeze aligns theta with the
    #    projected last dimension of X.
    theta = theta.unsqueeze(-1)
    return torch.clamp(X - theta, min=0.0)


# /////////////////////////////////////////////////////////////////////////////////////////
#                             PHYSICS GRADIENT FUNCTIONS
# /////////////////////////////////////////////////////////////////////////////////////////
#
# B-only (no frequency dim), adapted from the 4-D vectorised gradients in
# PGA_models.py.  Conventions:
#   * Every returned gradient is an *ascent* direction (the conjugate-Wirtinger /
#     PyTorch ``z.grad`` convention).
#   * H is (B, N_antennas, N_users) -- the channel stays in antenna-first layout
#     as required by SelectionNet; the gradient functions internally transpose to
#     (B, N_users, N_antennas) where the comm physics expects "users-first".

# Normalised noise power (must match system_config.sigma2).
_SIGMA2 = 1.0


def get_grad_F_com(H: torch.Tensor, F: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Ascent gradient of the sum-rate term R(F_eff, W) w.r.t. F_eff.

    Parameters
    ----------
    H : (B, N_antennas, N_users) complex channel.
    F : (B, N_antennas, N_rf) complex effective analog precoder F_eff.
    W : (B, N_rf, N_users) complex digital precoder.

    Returns
    -------
    (B, N_antennas, N_rf) complex ascent gradient dR / d conj(F_eff).
    """
    H_u = H.transpose(-2, -1)                         # (B, N_users, N_antennas)
    F_H = F.conj().transpose(-2, -1)                  # (B, N_rf, N_antennas)
    W_H = W.conj().transpose(-2, -1)                  # (B, N_users, N_rf)
    V = W @ W_H                                        # (B, N_rf, N_rf)

    w_cols = W.permute(0, 2, 1).unsqueeze(-1)          # (B, N_users, N_rf, 1)
    V_m = w_cols @ w_cols.conj().transpose(-2, -1)     # (B, N_users, N_rf, N_rf)
    V_mk = V.unsqueeze(1) - V_m                        # (B, N_users, N_rf, N_rf)

    h = H_u.unsqueeze(-1)                              # (B, N_users, N_antennas, 1)
    Htilde = h @ h.conj().transpose(-2, -1)            # (B, N_users, N_antennas, N_antennas)

    FVF_H = F @ V @ F_H                                # (B, N_antennas, N_antennas)

    qf1 = (h.conj().transpose(-2, -1) @ FVF_H.unsqueeze(1) @ h).squeeze(-1).squeeze(-1)
    denom1 = math.log(2.0) * (qf1 + _SIGMA2)           # (B, N_users)

    FVmk = F.unsqueeze(1) @ V_mk                       # (B, N_users, N_antennas, N_rf)
    FVmkF_H = FVmk @ F_H.unsqueeze(1)                  # (B, N_users, N_antennas, N_antennas)
    qf2 = (h.conj().transpose(-2, -1) @ FVmkF_H @ h).squeeze(-1).squeeze(-1)
    denom2 = math.log(2.0) * (qf2 + _SIGMA2)           # (B, N_users)

    HtF = Htilde @ F.unsqueeze(1)                      # (B, N_users, N_antennas, N_rf)
    grad1 = HtF @ V.unsqueeze(1)  / (denom1.unsqueeze(-1).unsqueeze(-1) + 1e-4)
    grad2 = HtF @ V_mk            / (denom2.unsqueeze(-1).unsqueeze(-1) + 1e-4)

    grad_F = (grad1 - grad2).sum(dim=1)                # (B, N_antennas, N_rf)
    return grad_F


def get_grad_F_crb(F: torch.Tensor, W: torch.Tensor, M_matrix: torch.Tensor) -> torch.Tensor:
    """Ascent gradient of log(CRLB^-1) = log(FIM) + const w.r.t. F_eff.

    ``M_matrix`` should be the sensing Fisher-like matrix in *antenna* space,
    i.e. ``A_dot^H R_N_inv A_dot`` of shape ``(N_antennas, N_antennas)`` (or
    ``(B, N_antennas, N_antennas)``).  It is broadcast over the batch.
    """
    W_H = W.conj().transpose(-2, -1)                  # (B, N_users, N_rf)
    F_H = F.conj().transpose(-2, -1)                  # (B, N_rf, N_antennas)

    inner_mat = W_H @ F_H @ M_matrix @ F @ W           # (B, N_users, N_users)
    fim = torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1)  # (B,)

    numerator = M_matrix @ F @ W @ W_H                 # (B, N_antennas, N_rf)
    return numerator / fim.unsqueeze(-1).unsqueeze(-1)


def get_grad_W_com(H: torch.Tensor, F: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Ascent gradient of the sum-rate term R(F_eff, W) w.r.t. W.

    Returns
    -------
    (B, N_rf, N_users) complex ascent gradient dR / d conj(W).
    """
    H_u = H.transpose(-2, -1)                         # (B, N_users, N_antennas)
    F_H = F.conj().transpose(-2, -1)                  # (B, N_rf, N_antennas)
    W_H = W.conj().transpose(-2, -1)                  # (B, N_users, N_rf)
    V = W @ W_H                                        # (B, N_rf, N_rf)
    N_users = W.shape[-1]
    grad_W = torch.zeros_like(W)

    for m in range(N_users):
        h_m = H_u[:, m, :].unsqueeze(-1)               # (B, N_antennas, 1)
        Htilde_m = h_m @ h_m.conj().transpose(-2, -1)  # (B, N_antennas, N_antennas)
        Hbar_m = F_H @ Htilde_m @ F                    # (B, N_rf, N_rf)
        denom = math.log(2.0) * (
            torch.diagonal(W @ W_H @ Hbar_m, dim1=-2, dim2=-1).sum(-1) + _SIGMA2
        )                                               # (B,)
        grad_m = Hbar_m @ W / denom.unsqueeze(-1).unsqueeze(-1)  # (B, N_rf, N_users)
        # Keep only the m-th user's column (the per-user ZF gradient).
        mask_m = torch.zeros_like(W)
        mask_m[:, :, m] = 1.0
        grad_W = grad_W + grad_m * mask_m

    return grad_W


def get_grad_W_crb(F: torch.Tensor, W: torch.Tensor, M_matrix: torch.Tensor) -> torch.Tensor:
    """Ascent gradient of log(CRLB^-1) w.r.t. W.

    Returns
    -------
    (B, N_rf, N_users) complex ascent gradient.
    """
    F_H = F.conj().transpose(-2, -1)                  # (B, N_rf, N_antennas)
    W_H = W.conj().transpose(-2, -1)                  # (B, N_users, N_rf)

    inner_mat = W_H @ F_H @ M_matrix @ F @ W           # (B, N_users, N_users)
    fim = torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1)  # (B,)

    numerator = F_H @ M_matrix @ F @ W                 # (B, N_rf, N_users)
    return numerator / fim.unsqueeze(-1).unsqueeze(-1)


# /////////////////////////////////////////////////////////////////////////////////////////
#                             TASK 2: FULL UNFOLDED NETWORK
# /////////////////////////////////////////////////////////////////////////////////////////

def build_fixed_subconnected_mask(n_antennas: int, n_rf_chains: int) -> torch.Tensor:
    """Build the fixed block sub-connected mask (N_antennas, N_rf).

    Every RF chain is connected to a contiguous block of
    ``n_antennas // n_rf_chains`` antennas, so each row has exactly one 1 and
    the rows sum to one.  Used as the S_0 initialisation of the ``fixed``
    variant of JointUPGANet.
    """
    if n_antennas % n_rf_chains != 0:
        raise ValueError(
            f"n_antennas ({n_antennas}) must be divisible by n_rf_chains ({n_rf_chains})"
        )
    mask = torch.zeros(n_antennas, n_rf_chains)
    antennas_per_rf = n_antennas // n_rf_chains
    for r in range(n_rf_chains):
        mask[r * antennas_per_rf:(r + 1) * antennas_per_rf, r] = 1.0
    return mask


class JointUPGANet(nn.Module):
    """Joint deep-unfolding PGA, structured like ``PGA_models.PGA_Unfold_JX``.

    A single module that runs the whole unfolded projected-gradient-ascent in
    ``execute_PGA`` (mirroring ``PGA_Unfold_JX.execute_PGA``): the connection
    matrix ``S`` is optimised jointly with the analog precoder ``F`` and the
    digital precoder ``W`` inside the same unfolded iterations.

    The initial connection matrix ``S_0`` is either produced by ``SelectionNet``
    from the channel and sensing direction (``s_init='selection'``) or taken as
    the fixed block sub-connected mask (``s_init='fixed'``, every RF chain serves
    ``N_antennas / N_rf`` antennas).

    Learnable parameters
    ---------------------
    step_size : (J, I, 3) per-inner-step step sizes, exactly like the
        ``step_size`` tensor of ``PGA_Unfold_JX`` (``[J, I, K+1]``) but with a
        third column for the connection matrix S.  Columns:
            step_size[j, ii, 0] : step size for F at inner step j, outer iter ii.
            step_size[0, ii, 1] : step size for S (once per outer iteration).
            step_size[0, ii, 2] : step size for W (once per outer iteration).
    """

    def __init__(self, step_size: torch.Tensor, n_antennas: int, n_rf_chains: int, n_users: int,
                 s_init: str = "fixed",
    ) -> None:
        super().__init__()

        assert s_init in ("selection", "fixed"), \
            f"s_init must be 'selection' or 'fixed', got {s_init!r}"

        self.step_size = nn.Parameter(step_size.clone())   # (J, I, 3): (F, S, W) step sizes
        self.n_antennas = n_antennas
        self.n_rf_chains = n_rf_chains
        self.n_users = n_users
        self.s_init = s_init

        self.selection_net = SelectionNet(n_antennas=n_antennas, n_rf_chains=n_rf_chains, n_users=n_users)

        if s_init == "fixed":
            self.register_buffer("fixed_s0", build_fixed_subconnected_mask(n_antennas, n_rf_chains))

    def execute_PGA(self, H: torch.Tensor, psi0: torch.Tensor, M_matrix: torch.Tensor, omega: float,
                    Pt, n_iter_outer: int, n_iter_inner: int, xi_0,
                    tau: float = 1.0, hard: bool = False, track_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full unfolded PGA (mirrors ``PGA_Unfold_JX.execute_PGA``).

        Parameters
        ----------
        H : (B, N_antennas, N_users) complex channel.
        psi0 : (B,) sensing direction (feeding SelectionNet when ``s_init='selection'``).
        M_matrix : (B, N_antennas, N_antennas) Hermitian PSD sensing Fisher-like
            matrix (``A_dot^H R_N_inv A_dot``).
        omega : float comm/sensing weighting.
        Pt : scalar or (B,) transmit power budget.
        n_iter_outer : int number of outer iterations I.
        n_iter_inner : int number of inner steps J per outer iteration.
        xi_0 : sensing reference amplitude (for the CRLB metric).
        tau : Gumbel-softmax temperature for SelectionNet.
        hard : straight-through one-hot mode for the final S.
        track_metrics : whether to record per-iteration rate / CRLB.

        Returns
        -------
        rate_over_iters, crb_over_iters : (n_iter_outer,) per-iteration metrics
            (zeros if ``track_metrics`` is False).
        F, S, W : final (F, S, W) after all outer iterations.
        """
        B = H.shape[0]

        # ---- Initial connection matrix S_0 ------------------------------------
        if self.s_init == "fixed":
            S = self.fixed_s0.expand(B, -1, -1).clone()          # (B, Nt, Nrf)
        else:
            S0, _ = self.selection_net(H, psi0, tau=tau, hard=hard)
            S = project_to_simplex_rows(S0)                      # defensive re-projection

        # ---- Initial (F0, W0) from the channel (like PGA_Unfold_JX.initialize) --
        F, W = initialize_joint(H, Pt, self.n_rf_chains)

        rate_over_iters = torch.zeros(n_iter_outer, device=H.device)
        crb_over_iters = torch.zeros(n_iter_outer, device=H.device)

        for ii in range(n_iter_outer):
            # ---- Inner loop: J steps of F ascent (S held fixed) --------------
            F_hat = F.clone()
            S_hat = S.clone()
            for j in range(n_iter_inner):
                F_eff = F_hat * S_hat                            # elementwise (Hadamard)
                grad_F_eff = omega * get_grad_F_com(H, F_eff, W) + get_grad_F_crb(F_eff, W, M_matrix)

                grad_F = S_hat * grad_F_eff                      # complex * real
                grad_S = 2 * torch.real(torch.conj(F_hat) * grad_F_eff)  # real

                F_hat = F_hat + self.step_size[j, ii, 0] * grad_F

                F_hat = F_hat / F_hat.abs().clamp_min(1e-8)

                S_hat = S_hat + self.step_size[j, ii, 1] * grad_S
                S_hat = project_to_simplex_rows(S_hat)


            F, S = F_hat, S_hat

            # ---- W update (once per outer iteration) --------------------------
            F_eff_new = F * S
            grad_W = omega * get_grad_W_com(H, F_eff_new, W) + get_grad_W_crb(F_eff_new, W, M_matrix)
            W = W + self.step_size[0, ii, 2] * grad_W

            # Power projection: ||F_eff_new W||_F^2 == Pt (per sample).
            prod = F_eff_new @ W                                 # (B, N_antennas, N_users)
            fro_norm = torch.linalg.matrix_norm(prod, dim=(-2, -1)).clamp_min(1e-8)  # (B,)
            P_BS_vec = torch.as_tensor(Pt, device=W.device, dtype=W.real.dtype)
            scale = torch.sqrt(P_BS_vec) / fro_norm              # (B,) or scalar
            while scale.dim() < W.dim():
                scale = scale.unsqueeze(-1)
            W = scale * W

            # ---- Per-iteration metrics (hard mask for the reported objective) --
            if track_metrics:
                S_metric = S
                if hard:
                    winners = S.argmax(dim=-1)                   # (B, Nt)
                    S_hard = torch.zeros_like(S)
                    S_hard.scatter_(-1, winners.unsqueeze(-1), 1.0)
                    S_metric = S_hard
                F_eff_m = F * S_metric
                rate_over_iters[ii] = get_sum_rate_joint(H, F_eff_m, W, Pt).detach()
                crb_over_iters[ii] = torch.mean(get_crb_joint(F_eff_m, W, M_matrix, xi_0, Pt)).detach()

        # ---- Final hard mask (straight-through estimator) ---------------------
        # In hard mode S already carries the hard one-hot forward value from the
        # inner loop; this final STE just re-asserts the discrete value on the
        # returned tensor (gradient still flows through the soft simplex S).
#        if hard:
#            winners = S.argmax(dim=-1)                           # (B, Nt)
#            S_hard = torch.zeros_like(S)
#            S_hard.scatter_(-1, winners.unsqueeze(-1), 1.0)
#            S = S_hard - S.detach() + S                          # STE: hard value, soft grad

        return rate_over_iters, crb_over_iters, F, S, W


# /////////////////////////////////////////////////////////////////////////////////////////
#                             EVALUATION / LOSS HELPERS
# /////////////////////////////////////////////////////////////////////////////////////////

def normalize_power_joint(F_eff: torch.Tensor, W: torch.Tensor, Pt) -> torch.Tensor:
    """Rescale W so ``||F_eff @ W||_F^2 == Pt`` per sample.

    F_eff is left untouched (its unit-modulus structure times the row-stochastic
    S mask must be preserved, matching ``skip_unit_modulus=True`` in the legacy
    ``normalize``).  ``Pt`` may be a scalar or a per-sample ``(B,)`` tensor.
    """
    power = torch.linalg.matrix_norm(F_eff @ W, dim=(-2, -1)) ** 2  # (B,)
    power = power.clamp_min(1e-6)
    Pt_vec = torch.as_tensor(Pt, device=W.device, dtype=power.real.dtype)
    scale = torch.sqrt(Pt_vec) / torch.sqrt(power)
    return scale.view(-1, 1, 1) * W


def get_sum_rate_joint(H: torch.Tensor, F_eff: torch.Tensor, W: torch.Tensor, Pt, skip_unit_modulus: bool = True,
) -> torch.Tensor:
    """Scalar sum rate for the effective (masked) precoder F_eff = F * S.

    H is (B, N_antennas, N_users); F_eff (B, N_antennas, N_rf); W (B, N_rf,
    N_users).  Mirrors ``utility.get_sum_rate`` for K=1.
    """
    if not skip_unit_modulus:
        F_eff = F_eff / (F_eff.abs() + 1e-12)
    W = normalize_power_joint(F_eff, W, Pt)

    H_u = H.transpose(-2, -1)                          # (B, N_users, N_antennas)
    F_H = F_eff.conj().transpose(-2, -1)
    W_H = W.conj().transpose(-2, -1)
    V = W @ W_H                                        # (B, N_rf, N_rf)
    N_users = W.shape[-1]

    mask = (1 - torch.eye(N_users, device=W.device, dtype=W.dtype))
    W_m_all = W.unsqueeze(1) * mask.view(1, N_users, 1, N_users)  # (B, U, N_rf, U)
    V_m_all = W_m_all @ W_m_all.conj().transpose(-1, -2)          # (B, U, N_rf, N_rf)

    h = H_u.unsqueeze(-1)                              # (B, U, N_antennas, 1)
    Htilde = h @ h.conj().transpose(-1, -2)            # (B, U, N_antennas, N_antennas)

    FVF_H = F_eff @ V @ F_H                            # (B, N_antennas, N_antennas)
    trace_1 = (FVF_H.unsqueeze(1) @ Htilde).diagonal(dim1=-1, dim2=-2).sum(-1)  # (B, U)

    FVmFH = F_eff.unsqueeze(1) @ V_m_all @ F_H.unsqueeze(1)          # (B, U, N_ant, N_ant)
    trace_2 = (FVmFH @ Htilde).diagonal(dim1=-1, dim2=-2).sum(-1)   # (B, U)

    rate = (torch.log2(trace_1 + _SIGMA2) - torch.log2(trace_2 + _SIGMA2)).real.sum(-1)  # (B,)
    return rate.mean()


def get_crb_joint(F_eff: torch.Tensor, W: torch.Tensor, M_matrix: torch.Tensor, xi_0, Pt = None,
) -> torch.Tensor:
    """Per-sample ``log(CRLB^-1)`` for the effective precoder F_eff.

    Returns a ``(B,)`` tensor equal to ``log(FIM) + log(2 xi_0^2)``.
    If ``Pt`` is provided, W is first power-normalized.
    """
    if Pt is not None:
        W = normalize_power_joint(F_eff, W, Pt)
    W_H = W.conj().transpose(-2, -1)
    F_H = F_eff.conj().transpose(-2, -1)
    inner_mat = W_H @ F_H @ M_matrix @ F_eff @ W       # (B, N_users, N_users)
    fim = torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1).real  # (B,)
    const = math.log(2.0 * xi_0 ** 2)
    return torch.log(fim + 1e-12) + const


def get_joint_loss(F: torch.Tensor, S: torch.Tensor, W: torch.Tensor, H: torch.Tensor, M_matrix: torch.Tensor, omega: float, xi_0, Pt,
) -> torch.Tensor:
    """Scalar unsupervised loss = -(omega * R + mean(log CRLB^-1))."""
    F_eff = F * S
    rate = get_sum_rate_joint(H, F_eff, W, Pt)
    crb = torch.mean(get_crb_joint(F_eff, W, M_matrix, xi_0, Pt))
    return -(omega * rate + crb)


def load_joint_state_dict(model: nn.Module, state: dict, n_inner: int) -> None:
    """Load a JointUPGANet state_dict, remapping the legacy step-size keys.

    Older checkpoints stored the per-layer step sizes as three separate
    parameters ``layers.<i>.mu`` (J,), ``layers.<i>.kappa`` (J,) and
    ``layers.<i>.lambda_`` (scalar).  The current model instead stores a single
    ``step_size`` tensor of shape ``(J, I, 3)`` with columns ``(F, S, W)``.
    This helper detects the legacy layout and rebuilds the full ``step_size``
    tensor from the per-layer ``mu`` / ``kappa`` / ``lambda_`` so old
    checkpoints load without retraining.

    Parameters
    ----------
    model : nn.Module
        The ``JointUPGANet`` instance to load into.
    state : dict
        The raw ``state_dict`` (e.g. from ``torch.load``).
    n_inner : int
        Number of inner steps J (used to size the ``mu`` column).
    """
    # Detect the legacy layout: any key ending in ".mu".
    if any(k.endswith(".mu") for k in state):
        # Collect per-layer legacy step sizes.
        layers = {}
        for k, v in state.items():
            if k.endswith(".mu"):
                prefix = k[:-len(".mu")]
                layers[prefix] = {
                    "mu": v,
                    "kappa": state.get(prefix + ".kappa", torch.tensor(1e-2)),
                    "lambda_": state.get(prefix + ".lambda_", torch.tensor(1e-2)),
                }

        # Rebuild the (J, I, 3) step_size tensor from the per-layer values.
        n_outer = len(layers)
        step = torch.zeros(n_inner, n_outer, 3,
                           dtype=next(iter(layers.values()))["mu"].dtype,
                           device=next(iter(layers.values()))["mu"].device)
        for ii, prefix in enumerate(sorted(layers, key=lambda p: int(p.rsplit(".", 1)[-1]))):
            mu = layers[prefix]["mu"]
            kappa = layers[prefix]["kappa"]
            lam = layers[prefix]["lambda_"]
            step[:, ii, 0] = mu
            step[0, ii, 1] = kappa.reshape(-1)[0]
            step[0, ii, 2] = lam.reshape(-1)[0]

        new_state = {k: v for k, v in state.items()
                     if not (k.endswith(".mu") or k.endswith(".kappa") or k.endswith(".lambda_"))}
        new_state["step_size"] = step
        state = new_state
    model.load_state_dict(state, strict=False)


def initialize_joint(H: torch.Tensor, Pt, n_rf_chains: int):
    """Initial (F0, W0) for the joint network.

    F0 is an SVD-based unit-modulus analog precoder (the top ``n_rf_chains``
    right-singular vectors of the channel, matching the legacy ``init_scheme =
    'svd'`` used by the fixed sub-connected baseline); W0 is a ridge-ZF digital
    precoder matched to the effective channel ``H^T F0``, power-normalised.

    Using the SVD init (instead of a random unit-modulus F0) gives the joint
    network the same strong starting point as the fixed sub-connected baseline,
    so the comparison in ``main_iter_joint.py`` is fair and the joint model's
    mask optimisation is not handicapped by a weak initialisation.
    """
    B, N_antennas, _ = H.shape

    # SVD-based F0: top n_rf_chains right-singular vectors, unit modulus.
    # H is (B, N_antennas, N_users); svd of H^T (B, N_users, N_antennas) gives
    # V_H of shape (B, N_antennas, N_antennas) whose leading rows are the
    # dominant right-singular vectors (matching legacy init_scheme='svd').
    _, _, V_H = torch.linalg.svd(H.transpose(-2, -1))
    F0 = V_H[:, :n_rf_chains, :].transpose(-2, -1)     # (B, N_antennas, N_rf)
    F0 = F0 / F0.abs().clamp_min(1e-8)

    H_u = H.transpose(-2, -1)                          # (B, N_users, N_antennas)
    H_eff = H_u @ F0                                   # (B, N_users, N_rf)
    G = H_eff @ H_eff.conj().transpose(-1, -2)         # (B, N_users, N_users)
    lam = 1e-2 * torch.diagonal(G, dim1=-2, dim2=-1).real.mean().detach()
    I_u = torch.eye(H_eff.shape[1], device=H.device, dtype=H.dtype)
    W0 = H_eff.conj().transpose(-1, -2) @ torch.linalg.inv(G + lam.view(-1, 1, 1) * I_u)

    W0 = normalize_power_joint(F0, W0, Pt)
    return F0, W0


if __name__ == "__main__":
    # Minimal smoke test for the simplex projection (the layer/network cannot run
    # until the gradient stubs are implemented).
    torch.manual_seed(0)
    X = torch.randn(3, 4, 6)
    P = project_to_simplex_rows(X)
    row_sums = P.sum(dim=-1)
    print(f"projected shape: {tuple(P.shape)}")
    print(f"row sums min={row_sums.min():.6f} max={row_sums.max():.6f}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert (P >= 0).all()
    print("project_to_simplex_rows smoke test passed.")
