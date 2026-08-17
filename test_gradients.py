"""Gradient-correctness check for the joint unfolded PGA chain rule.

Verifies the two hand-derived formulas used inside ``JointUnfoldedLayer``::

    grad_F = S * grad_F_eff
    grad_S = real(conj(F) * grad_F_eff)

where ``F_eff = F * S`` (Hadamard) and ``grad_F_eff`` is the objective gradient
w.r.t. the effective precoder.  These are the *only* formulas under test: the
physics (R, CRLB) is replaced below by simple, differentiable, dimensionally
consistent placeholders, and ``grad_F_eff`` is built from the corresponding
analytic gradients exactly as the real ``get_grad_F_com`` / ``get_grad_F_crb``
stubs would.

Run::

    python test_gradients.py
"""

from __future__ import annotations

import torch

from joint_upganet import project_to_simplex_rows

EPS = 1e-6


# --------------------------------------------------------------------------------------
# TEST-ONLY SIMPLIFIED IMPLEMENTATION (do not reuse for real physics)
# --------------------------------------------------------------------------------------
# These stand in for the real comm / CRLB objective terms and their gradients so the
# chain rule can be checked in isolation.  They are self-consistent (each analytic
# gradient below is the exact ascent gradient of the matching scalar term), but are
# NOT the physical sum-rate / CRLB of the mmWave ISAC system.
#
#   R(F_eff, W, H)            = Re( sum_{b,n,k} conj(H) * (F_eff @ W) )
#   log(CRLB^-1)(F_eff, W, M) = log( Re( trace(F_eff^H F_eff M) ) )
# --------------------------------------------------------------------------------------

def _R(F_eff: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Scalar comm-like term (real)."""
    return torch.real(torch.sum(H.conj() * (F_eff @ W)))


def _log_crlb_inv(F_eff: torch.Tensor, W: torch.Tensor, M_matrix: torch.Tensor) -> torch.Tensor:
    """Scalar sensing-like term (real): log of a Fisher-information trace."""
    F_eff_H = F_eff.conj().transpose(-1, -2)
    fim = (F_eff_H @ F_eff @ M_matrix).diagonal(dim1=-2, dim2=-1).sum(-1).real  # (B,)
    return torch.log(fim + EPS).sum()


def _grad_F_com(H: torch.Tensor, F_eff: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Analytic ascent gradient of ``_R`` w.r.t. F_eff (d/d conj(F_eff)).

    Verified against ``torch.autograd.grad``; PyTorch's complex backward through
    ``torch.real`` passes the gradient with unit factor, so there is no 1/2.
    """
    return H @ W.conj().transpose(-1, -2)


def _grad_F_crb(F_eff: torch.Tensor, W: torch.Tensor, M_matrix: torch.Tensor) -> torch.Tensor:
    """Analytic ascent gradient of ``_log_crlb_inv`` w.r.t. F_eff (d/d conj(F_eff)).

    Verified against ``torch.autograd.grad``; the factor 2 comes from the
    quadratic trace ``trace(F^H F M)`` being paired with PyTorch's ``real``
    backward (unit factor rather than the Wirtinger 1/2).
    """
    F_eff_H = F_eff.conj().transpose(-1, -2)
    fim = (F_eff_H @ F_eff @ M_matrix).diagonal(dim1=-2, dim2=-1).sum(-1).real  # (B,)
    return 2.0 * (F_eff @ M_matrix) / (fim + EPS).unsqueeze(-1).unsqueeze(-1)


def _randn_complex(*shape, device=None):
    return torch.randn(*shape, device=device, dtype=torch.complex64)


def _hermitian_psd(n: int, device=None) -> torch.Tensor:
    """Random n x n Hermitian positive-definite matrix via A A^H + eps*I."""
    A = torch.randn(n, n, dtype=torch.complex64, device=device)
    return A @ A.conj().transpose(-1, -2) + EPS * torch.eye(n, dtype=torch.complex64, device=device)


def main() -> None:
    torch.manual_seed(0)

    # Problem size per the spec: N=4 antennas, M=2 RF chains, K=3 users.
    N, M, K = 4, 2, 3
    B = 2  # small batch so batching is exercised (use B=1 for a single sample)

    omega = 0.5

    F = _randn_complex(B, N, M)
    W = _randn_complex(B, M, K)
    H = _randn_complex(B, N, K)
    M_matrix = _hermitian_psd(M).unsqueeze(0).expand(B, -1, -1).contiguous()

    # Row-stochastic S (softmax over the RF-chain axis, entries in [0, 1]).
    logits = torch.randn(B, N, M)
    S = torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    # Path A: hand-derived chain rule (the formulas under test)
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        F_eff = F * S
        grad_F_eff = omega * _grad_F_com(H, F_eff, W) + _grad_F_crb(F_eff, W, M_matrix)
        grad_F_hand = S * grad_F_eff
        grad_S_hand = torch.real(torch.conj(F) * grad_F_eff)

    # ------------------------------------------------------------------ #
    # Path B: torch.autograd on the same scalar objective
    # ------------------------------------------------------------------ #
    F.requires_grad_(True)
    S.requires_grad_(True)

    F_eff = F * S
    g = omega * _R(F_eff, W, H) + _log_crlb_inv(F_eff, W, M_matrix)
    g.backward()

    grad_F_auto = F.grad
    grad_S_auto = S.grad

    # ------------------------------------------------------------------ #
    # Compare
    # ------------------------------------------------------------------ #
    diff_F = (grad_F_hand - grad_F_auto).abs().max().item()
    diff_S = (grad_S_hand - grad_S_auto).abs().max().item()
    atol = 1e-4

    ok_F = bool(torch.allclose(grad_F_hand, grad_F_auto, atol=atol, rtol=1e-4))
    ok_S = bool(torch.allclose(grad_S_hand, grad_S_auto, atol=atol, rtol=1e-4))

    print("=" * 60)
    print("Joint unfolded PGA -- chain-rule gradient check")
    print("=" * 60)
    print(f"grad_F  : max abs diff = {diff_F:.3e}   {'PASS' if ok_F else 'FAIL'}")
    print(f"grad_S  : max abs diff = {diff_S:.3e}   {'PASS' if ok_S else 'FAIL'}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Bonus: verify the simplex projection (Task 1) on a random batch
    # ------------------------------------------------------------------ #
    X = torch.randn(B, N, M)
    P = project_to_simplex_rows(X)
    row_sums = P.sum(dim=-1)
    ok_proj = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)) and bool((P >= -1e-8).all())
    print(f"simplex : row sums max err = {(row_sums - 1.0).abs().max().item():.3e}   "
          f"{'PASS' if ok_proj else 'FAIL'}")

    if ok_F and ok_S and ok_proj:
        print("\nAll gradient checks passed.")
    else:
        print("\nSome checks FAILED -- see above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
