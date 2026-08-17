"""Evaluate the trained JointUPGANet vs SNR.

Sweeps the SNR list, runs the full joint unfolding, and plots
  (a) the objective J = omega*R + log(CRLB^-1) vs SNR,
  (b) the sum-rate R vs SNR,
  (c) the CRLB vs SNR.

Run:  python main_SNR_joint.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from system_config import *
from utility import get_data_tensor, safe_legend
from joint_upganet import (
    JointUPGANet, get_sum_rate_joint, get_crb_joint, initialize_joint,
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

    _, H_test0 = get_data_tensor(data_source)
    H_test = H_test0[:, :test_size, :, :]                     # (K, B, M, Nt)
    B_test = H_test.shape[1]
    print(f"H_test (K, B, M, Nt): {tuple(H_test.shape)}")

    M_matrix = (A_dot.conj().T @ R_N_inv @ A_dot).to(H_test.device)  # (Nt, Nt)

    model = JointUPGANet(
        n_outer=N_OUTER, n_inner=N_INNER,
        n_antennas=Nt, n_rf_chains=Nrf, n_users=M,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    H_joint = to_joint_channel(H_test).to(device)             # (B, Nt, M)
    psi0 = torch.full((B_test,), desired_angle_rad_torch, device=device)

    obj = np.zeros(len(snr_dB_list))
    rate = np.zeros(len(snr_dB_list))
    crlb = np.zeros(len(snr_dB_list))

    for ss, snr_dB in enumerate(snr_dB_list):
        snr_ss = 10 ** (snr_dB / 10)
        snr_t = torch.full((B_test,), snr_ss, dtype=torch.float32, device=device)
        print(f'Evaluating JointUPGANet at SNR = {snr_dB} dB ...')

        with torch.no_grad():
            F0, W0 = initialize_joint(H_joint, snr_t, Nrf)
            F, S, W = model(F0, W0, H_joint, psi0, M_matrix, OMEGA, snr_t,
                            tau=0.05, hard=True)
            F_eff = F * S
            r = get_sum_rate_joint(H_joint, F_eff, W, snr_t)
            c = torch.mean(get_crb_joint(F_eff, W, M_matrix, xi_0, snr_t))

            rate[ss] = r.item()
            crlb[ss] = float(np.exp(-c.item()))          # CRLB = 1 / FIM
            obj[ss] = OMEGA * r.item() + c.item()

        print(f'  R = {rate[ss]:.4f}, log(CRLB^-1) = {np.log(1.0 / crlb[ss]):.4f}, '
              f'J = {obj[ss]:.4f}')

    # ---- Objective vs SNR
    plt.figure(figsize=(8, 4.5))
    plt.plot(snr_dB_list, obj, '-o', color='red', linewidth=3, markersize=8,
             label='JointUPGANet')
    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel(r'$J = \omega R + \log(\mathrm{CRLB}^{-1})$', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + f'JointUPGANet_obj_vs_SNR_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    # ---- Rate vs SNR
    plt.figure(figsize=(8, 4.5))
    plt.plot(snr_dB_list, rate, '-o', color='blue', linewidth=3, markersize=8,
             label='JointUPGANet')
    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + f'JointUPGANet_rate_vs_SNR_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    # ---- CRLB vs SNR
    plt.figure(figsize=(8, 4.5))
    plt.plot(snr_dB_list, crlb, '-o', color='green', linewidth=3, markersize=8,
             label='JointUPGANet')
    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel(r'$\mathrm{CRLB}$', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + f'JointUPGANet_CRB_vs_SNR_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    print(f'Saved figures to {directory_result}')


if __name__ == "__main__":
    main()
