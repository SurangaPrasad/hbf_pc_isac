"""Evaluate the trained JointUPGANet vs SNR, against the core baselines.

Sweeps the SNR list and plots, for the objective J = omega*R + log(CRLB^-1),
the sum-rate R, and the CRLB:

  1. JointUPGANet        - joint deep-unfolding of (F, S, W)  [ours]
  2. Full-connected HBF  - frozen UPGA F, no connectivity mask
  3. Fixed sub-connected - frozen UPGA F gated by the uniform block mask
  4. Adaptive connected  - frozen UPGA F gated by the trained SelectionNet mask

Baselines 2-4 follow the exact protocol of ``main_selection.py``:
``plot_selectionnet_objective_vs_snr`` (frozen UPGA J5 beamformer, W re-derived
with ``compute_digital_precoder`` for the masked effective channel,
``skip_unit_modulus=True``).

Run:  python main_SNR_joint.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from system_config import *
from utility import (
    get_data_tensor, safe_legend, compute_digital_precoder,
    generage_partial_connection_mask, get_sum_rate, get_crb_fe,
)
from SelectionNet import SelectionNet
from main_selection import load_pretrained_upga, REDERIVE_DIGITAL_W
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

    # -- 1. Joint model -----------------------------------------------------
    model = JointUPGANet(
        n_outer=N_OUTER, n_inner=N_INNER,
        n_antennas=Nt, n_rf_chains=Nrf, n_users=M,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    H_joint = to_joint_channel(H_test).to(device)             # (B, Nt, M)
    psi0 = torch.full((B_test,), desired_angle_rad_torch, device=device)

    # -- 2-4. Baselines (frozen UPGA beamformer + connectivity masks) -------
    upga = load_pretrained_upga(model_file_name_UPGA_J5, n_iter_inner_J5, device)
    selnet = SelectionNet(n_antennas=Nt, n_rf_chains=Nrf, n_users=M).to(device)
    selnet.load_state_dict(torch.load(
        directory_model + f'SelectionNet_J{n_iter_inner_J5}.pth', map_location=device))
    selnet.eval()

    fixed_mask = generage_partial_connection_mask(Nt, Nrf).real.to(device)  # (Nt, Nrf)
    H_sel = H_test[0].transpose(1, 2)                          # (B, Nt, M)
    with torch.no_grad():
        S_hard, _ = selnet(H_sel, psi0, tau=0.05, hard=True)   # (B, Nt, Nrf)

    def baseline_metrics(F_eff, W, snr_ss):
        """Objective / rate / CRB for a given (F_eff, W) using the legacy physics.

        Note: this does NOT re-derive W.  The caller decides which W to use (the
        UPGA's own optimized W for the full-connected case, or a matched W for a
        masked array).  Re-deriving W for the *full-connected* array with
        ``compute_digital_precoder`` (ridge-ZF) discards the UPGA's optimized W
        and makes the full-connected curve look artificially low.
        """
        rate = get_sum_rate(H_test, F_eff, W, snr_ss, skip_unit_modulus=True)
        crb = torch.mean(get_crb_fe(H_test, F_eff, W, xi_0, A_dot, R_N_inv,
                                    snr_ss, skip_unit_modulus=True))
        return OMEGA * rate + crb, rate, crb

    obj      = np.zeros(len(snr_dB_list))   # JointUPGANet
    rate     = np.zeros(len(snr_dB_list))
    crlb     = np.zeros(len(snr_dB_list))
    obj_full = np.zeros(len(snr_dB_list))   # Full-connected
    rate_full = np.zeros(len(snr_dB_list))
    crlb_full = np.zeros(len(snr_dB_list))
    obj_sub  = np.zeros(len(snr_dB_list))   # Fixed sub-connected
    rate_sub = np.zeros(len(snr_dB_list))
    crlb_sub = np.zeros(len(snr_dB_list))
    obj_sel  = np.zeros(len(snr_dB_list))   # Adaptive (SelectionNet)
    rate_sel = np.zeros(len(snr_dB_list))
    crlb_sel = np.zeros(len(snr_dB_list))

    for ss, snr_dB in enumerate(snr_dB_list):
        snr_ss = 10 ** (snr_dB / 10)
        snr_t = torch.full((B_test,), snr_ss, dtype=torch.float32, device=device)
        print(f'Evaluating at SNR = {snr_dB} dB ...')

        with torch.no_grad():
            # --- JointUPGANet
            F0, W0 = initialize_joint(H_joint, snr_t, Nrf)
            F, S, W = model(F0, W0, H_joint, psi0, M_matrix, OMEGA, snr_t,
                            tau=0.05, hard=True)
            F_eff = F * S
            r = get_sum_rate_joint(H_joint, F_eff, W, snr_t)
            c = torch.mean(get_crb_joint(F_eff, W, M_matrix, xi_0, snr_t))
            rate[ss] = r.item()
            crlb[ss] = float(np.exp(-c.item()))
            obj[ss] = OMEGA * r.item() + c.item()

            # --- Baselines: frozen UPGA gives F (K, B, Nt, Nrf), W
            _, _, F_up, W_up, _, _ = upga.execute_PGA(
                H_test, xi_0, A_dot, R_N_inv,
                torch.tensor(snr_ss, dtype=torch.float32, device=device),
                n_iter_outer, n_iter_inner_J5, track_metrics=False)

            # Full-connected: use the UPGA's own optimized W (already matched to
            # the full-connected F).  This is the upper bound of the comparison.
            o_f, r_f, c_f = baseline_metrics(F_up, W_up, snr_ss)

            # Sub-connected: W_up is mismatched to the masked array, so re-derive
            # a matched digital precoder (ridge-ZF) for the masked F_eff.
            W_sub = compute_digital_precoder(H_test, F_up * fixed_mask) if REDERIVE_DIGITAL_W else W_up
            o_s, r_s, c_s = baseline_metrics(F_up * fixed_mask, W_sub, snr_ss)

            W_sel = compute_digital_precoder(H_test, F_up * S_hard.unsqueeze(0)) if REDERIVE_DIGITAL_W else W_up
            o_l, r_l, c_l = baseline_metrics(F_up * S_hard.unsqueeze(0), W_sel, snr_ss)

            obj_full[ss], rate_full[ss], crlb_full[ss] = o_f.item(), r_f.item(), float(np.exp(-c_f.item()))
            obj_sub[ss], rate_sub[ss], crlb_sub[ss] = o_s.item(), r_s.item(), float(np.exp(-c_s.item()))
            obj_sel[ss], rate_sel[ss], crlb_sel[ss] = o_l.item(), r_l.item(), float(np.exp(-c_l.item()))

        print(f'  Joint: J={obj[ss]:.4f} | Full: J={obj_full[ss]:.4f} | '
              f'Fixed Sub: J={obj_sub[ss]:.4f} | Adaptive: J={obj_sel[ss]:.4f}')

    # ---- Plot helpers -------------------------------------------------------
    cmap = {
        'JointUPGANet':  ('-o', 'red'),
        'Full-connected': ('-d', 'black'),
        'Fixed sub-connected': ('--s', 'blue'),
        'Adaptive (SelectionNet)': ('-.^', 'green'),
    }

    def plot_curves(snr, series, ylabel, fname):
        plt.figure(figsize=(8, 4.5))
        for label, (style, color) in cmap.items():
            plt.plot(snr, series[label], style, color=color, linewidth=3,
                     markersize=8, label=label)
        plt.xlabel('SNR [dB]', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid()
        safe_legend(loc='best', fontsize=11, labelspacing=0.15)
        plt.tight_layout()
        plt.savefig(directory_result + fname.format(Nt=Nt, OMEGA=OMEGA),
                    dpi=300, bbox_inches='tight', pad_inches=0.02)

    # ---- Objective vs SNR
    plot_curves(snr_dB_list, {
        'JointUPGANet': obj,
        'Full-connected': obj_full,
        'Fixed sub-connected': obj_sub,
        'Adaptive (SelectionNet)': obj_sel,
    }, r'$J = \omega R + \log(\mathrm{CRLB}^{-1})$',
        'JointUPGANet_obj_vs_SNR_{Nt}_{OMEGA}.png')

    # ---- Rate vs SNR
    plot_curves(snr_dB_list, {
        'JointUPGANet': rate,
        'Full-connected': rate_full,
        'Fixed sub-connected': rate_sub,
        'Adaptive (SelectionNet)': rate_sel,
    }, r'$R$ [bits/s/Hz]', 'JointUPGANet_rate_vs_SNR_{Nt}_{OMEGA}.png')

    # ---- CRLB vs SNR
    plot_curves(snr_dB_list, {
        'JointUPGANet': crlb,
        'Full-connected': crlb_full,
        'Fixed sub-connected': crlb_sub,
        'Adaptive (SelectionNet)': crlb_sel,
    }, r'$\mathrm{CRLB}$', 'JointUPGANet_CRB_vs_SNR_{Nt}_{OMEGA}.png')

    print(f'Saved figures to {directory_result}')


if __name__ == "__main__":
    main()
