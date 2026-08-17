"""Plot the JointUPGANet objective vs unfolded iterations (layers).

Runs the full joint unfolding once and records the physics objective
J = omega*R + log(CRLB^-1) after every outer iteration, so the convergence
behaviour of the learned step sizes can be inspected.

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

MODEL_FILE = directory_model + f'JointUPGANet_I{N_OUTER}_J{N_INNER}.pth'


def to_joint_channel(H_kb: torch.Tensor) -> torch.Tensor:
    """(K, B, M, Nt) -> (B, Nt, M) using the single frequency band."""
    return H_kb[0].transpose(1, 2)


def main():
    torch.manual_seed(3407)

    H_train, H_test0 = get_data_tensor(data_source)
    H_test = H_test0[:, :test_size, :, :]                     # (K, B, M, Nt)
    B_test = H_test.shape[1]

    M_matrix = (A_dot.conj().T @ R_N_inv @ A_dot).to(H_test.device)  # (Nt, Nt)

    model = JointUPGANet(
        n_outer=N_OUTER, n_inner=N_INNER,
        n_antennas=Nt, n_rf_chains=Nrf, n_users=M,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    H_joint = to_joint_channel(H_test).to(device)             # (B, Nt, M)
    psi0 = torch.full((B_test,), desired_angle_rad_torch, device=device)
    snr_t = torch.full((B_test,), snr, dtype=torch.float32, device=device)

    obj_iter = np.zeros(N_OUTER)
    rate_iter = np.zeros(N_OUTER)
    crlb_iter = np.zeros(N_OUTER)

    with torch.no_grad():
        F0, W0 = initialize_joint(H_joint, snr_t, Nrf)

        # ---- Produce S_0 (mirrors JointUPGANet.forward) then unroll manually.
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

    iter_x = np.arange(1, N_OUTER + 1)

    # ---- Objective vs iterations (the key convergence figure).
    plt.figure(figsize=(8, 5))
    plt.plot(iter_x, obj_iter, '-o', color='red', linewidth=3, markersize=7,
             label='JointUPGANet')
    plt.xlabel(r'Number of layers $(I)$', fontsize=14)
    plt.ylabel(r'$\omega R + \log(\mathrm{CRLB}^{-1})$', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + f'JointUPGANet_obj_vs_iter_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    # ---- Rate + log(CRLB^-1) vs iterations.
    plt.figure(figsize=(8, 5))
    plt.plot(iter_x, rate_iter, '-o', color='blue', linewidth=3, markersize=7,
             label='R')
    plt.plot(iter_x, crlb_iter, '-s', color='green', linewidth=3, markersize=7,
             label=r'$\log(\mathrm{CRLB}^{-1})$')
    plt.xlabel(r'Number of layers $(I)$', fontsize=14)
    plt.ylabel('Metric', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + f'JointUPGANet_metrics_vs_iter_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    print(f'Final (I={N_OUTER}): R={rate_iter[-1]:.4f}, '
          f'log(CRLB^-1)={crlb_iter[-1]:.4f}, J={obj_iter[-1]:.4f}')
    print(f'Saved figures to {directory_result}')


if __name__ == "__main__":
    main()
