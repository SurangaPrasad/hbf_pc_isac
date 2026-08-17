"""Plot the JointUPGANet objective vs unfolded iterations (layers).

Runs the full joint unfolding once (for both S_0 initialisation schemes) and
records the physics objective J = omega*R + log(CRLB^-1) after every outer
iteration, so the convergence behaviour of the learned step sizes can be
inspected.

Run:  python main_iter_joint.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from system_config import *
from utility import get_data_tensor, safe_legend
from joint_upganet import (
    JointUPGANet, project_to_simplex_rows, get_sum_rate_joint, get_crb_joint,
    initialize_joint,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

N_OUTER = n_iter_outer
N_INNER = n_iter_inner_J5


def joint_model_path(s_init: str) -> str:
    tag = "" if s_init == "selection" else f"_{s_init}"
    return directory_model + f'JointUPGANet{tag}_I{N_OUTER}_J{N_INNER}.pth'


def to_joint_channel(H_kb: torch.Tensor) -> torch.Tensor:
    """(K, B, M, Nt) -> (B, Nt, M) using the single frequency band."""
    return H_kb[0].transpose(1, 2)


def unroll_objective(s_init: str, H_joint, psi0, M_matrix, snr_t):
    """Run one variant once and record (obj, rate, crlb) per outer iteration."""
    model = JointUPGANet(
        n_outer=N_OUTER, n_inner=N_INNER,
        n_antennas=Nt, n_rf_chains=Nrf, n_users=M,
        s_init=s_init,
    ).to(device)
    model.load_state_dict(torch.load(joint_model_path(s_init), map_location=device))
    model.eval()

    obj_iter = np.zeros(N_OUTER)
    rate_iter = np.zeros(N_OUTER)
    crlb_iter = np.zeros(N_OUTER)

    with torch.no_grad():
        F0, W0 = initialize_joint(H_joint, snr_t, Nrf)

        # ---- Produce S_0 (mirrors JointUPGANet.forward) then unroll manually.
        if model.s_init == "fixed":
            S = model.fixed_s0.expand(H_joint.shape[0], -1, -1).clone()
        else:
            S0, _ = model.selection_net(H_joint, psi0, tau=0.05, hard=True)
            S = project_to_simplex_rows(S0)
        F, W = F0, W0

        for ii, layer in enumerate(model.layers):
            F, S, W = layer(F, S, W, H_joint, psi0, M_matrix, OMEGA, snr_t)
            F_eff = F * S
            r = get_sum_rate_joint(H_joint, F_eff, W, snr_t)
            c = torch.mean(get_crb_joint(F_eff, W, M_matrix, xi_0, snr_t))
            rate_iter[ii] = r.item()
            crlb_iter[ii] = c.item()
            obj_iter[ii] = OMEGA * r.item() + c.item()

    return obj_iter, rate_iter, crlb_iter


def main():
    torch.manual_seed(3407)

    _, H_test0 = get_data_tensor(data_source)
    H_test = H_test0[:, :test_size, :, :]                     # (K, B, M, Nt)
    B_test = H_test.shape[1]

    M_matrix = (A_dot.conj().T @ R_N_inv @ A_dot).to(H_test.device)  # (Nt, Nt)

    H_joint = to_joint_channel(H_test).to(device)             # (B, Nt, M)
    psi0 = torch.full((B_test,), desired_angle_rad_torch, device=device)
    snr_t = torch.full((B_test,), snr, dtype=torch.float32, device=device)

    obj_sel, rate_sel, crlb_sel = unroll_objective('selection', H_joint, psi0,
                                                   M_matrix, snr_t)
    obj_fix, rate_fix, crlb_fix = unroll_objective('fixed', H_joint, psi0,
                                                   M_matrix, snr_t)

    iter_x = np.arange(1, N_OUTER + 1)

    # ---- Objective vs iterations (the key convergence figure).
    plt.figure(figsize=(8, 5))
    plt.plot(iter_x, obj_sel, '-o', color='red', linewidth=3, markersize=6,
             label='JointUPGANet (selection init)')
    plt.plot(iter_x, obj_fix, '-^', color='purple', linewidth=3, markersize=6,
             label='JointUPGANet (fixed init)')
    plt.xlabel(r'Number of layers $(I)$', fontsize=14)
    plt.ylabel(r'$\omega R + \log(\mathrm{CRLB}^{-1})$', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + f'JointUPGANet_obj_vs_iter_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    # ---- Rate + log(CRLB^-1) vs iterations, for both variants.
    plt.figure(figsize=(8, 5))
    plt.plot(iter_x, rate_sel, '-o', color='red', linewidth=3, markersize=6,
             label='R (selection init)')
    plt.plot(iter_x, rate_fix, '-^', color='purple', linewidth=3, markersize=6,
             label='R (fixed init)')
    plt.plot(iter_x, crlb_sel, '--o', color='orange', linewidth=3, markersize=6,
             label=r'$\log(\mathrm{CRLB}^{-1})$ (selection init)')
    plt.plot(iter_x, crlb_fix, '--^', color='cyan', linewidth=3, markersize=6,
             label=r'$\log(\mathrm{CRLB}^{-1})$ (fixed init)')
    plt.xlabel(r'Number of layers $(I)$', fontsize=14)
    plt.ylabel('Metric', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(directory_result + f'JointUPGANet_metrics_vs_iter_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    print(f'Final (I={N_OUTER}):')
    print(f'  selection init: R={rate_sel[-1]:.4f}, '
          f'log(CRLB^-1)={crlb_sel[-1]:.4f}, J={obj_sel[-1]:.4f}')
    print(f'  fixed init:     R={rate_fix[-1]:.4f}, '
          f'log(CRLB^-1)={crlb_fix[-1]:.4f}, J={obj_fix[-1]:.4f}')
    print(f'Saved figures to {directory_result}')


if __name__ == "__main__":
    main()
