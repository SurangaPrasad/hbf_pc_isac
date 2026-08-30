"""Evaluate the trained JointUPGANet variants vs SNR, against the core baselines.

Sweeps the SNR list and plots, for the objective J = omega*R + log(CRLB^-1),
the sum-rate R, and the CRLB:

  1. JointUPGANet (selection init) - S_0 from SelectionNet        [ours]
  2. JointUPGANet (fixed init)     - S_0 from the fixed block mask [ours]
  3. Full-connected HBF            - frozen UPGA F, no mask
  4. Fixed sub-connected           - frozen UPGA F gated by the block mask
  5. Adaptive connected            - frozen UPGA F gated by the SelectionNet mask

Baselines 3-5 follow the exact protocol of ``main_selection.py``:
``plot_selectionnet_objective_vs_snr`` (frozen UPGA J5 beamformer, W re-derived
with ``compute_digital_precoder`` for the masked effective channel, except the
full-connected case which uses the UPGA's own optimized W).

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
    JointUPGANet, JointUPGANet_decay, get_sum_rate_joint, get_crb_joint,
    initialize_joint, load_joint_state_dict,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

N_OUTER = n_iter_outer
N_INNER = n_iter_inner_J5


def joint_model_path(s_init: str, decay: bool = False) -> str:
    tag = "" if s_init == "selection" else f"_{s_init}"
    if decay:
        return directory_model + f'JointUPGANet{tag}_decay_I{N_OUTER}_J{N_INNER}.pth'
    return directory_model + f'JointUPGANet{tag}_I{N_OUTER}_J{N_INNER}.pth'


def to_joint_channel(H_kb: torch.Tensor) -> torch.Tensor:
    """(K, B, M, Nt) -> (B, Nt, M) using the single frequency band."""
    return H_kb[0].transpose(1, 2)


def evaluate_joint(s_init: str, H_joint, psi0, M_matrix, snr_dB_list, B_test, decay: bool = False):
    """Run one JointUPGANet variant over the SNR list; return (obj, rate, crlb)."""
    if decay:
        model = JointUPGANet_decay(
            step_size=step_size_joint_decay,
            n_antennas=Nt, n_rf_chains=Nrf, n_users=M,
            s_init=s_init,
        ).to(device)
    else:
        model = JointUPGANet(
            step_size=step_size_joint,
            n_antennas=Nt, n_rf_chains=Nrf, n_users=M,
            s_init=s_init,
        ).to(device)
    load_joint_state_dict(model, torch.load(joint_model_path(s_init, decay), map_location=device), N_INNER)
    model.eval()

    obj = np.zeros(len(snr_dB_list))
    rate = np.zeros(len(snr_dB_list))
    crlb = np.zeros(len(snr_dB_list))

    for ss, snr_dB in enumerate(snr_dB_list):
        snr_ss = 10 ** (snr_dB / 10)
        snr_t = torch.full((B_test,), snr_ss, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, F, S, W = model.execute_PGA(
                H_joint, psi0, M_matrix, OMEGA, snr_t,
                N_OUTER, N_INNER, xi_0, tau=0.05, hard=True, track_metrics=False)
            # Binarize the connection matrix to the physically realizable
            # one-hot assignment before computing the metrics.  The soft
            # simplex S (rows like [0.7, 0.2, 0.1, 0.0]) gives F_eff fractional
            # gains on several RF chains -- more effective DoF/power than any
            # legal sub-connected hardware allows -- which inflates the
            # objective unfairly (it looked close to full-connected).
            winners = S.argmax(dim=-1)                       # (B, Nt)
            S_hard = torch.zeros_like(S)
            S_hard.scatter_(-1, winners.unsqueeze(-1), 1.0)
            F_eff = F * S_hard
            r = get_sum_rate_joint(H_joint, F_eff, W, snr_t)
            c = torch.mean(get_crb_joint(F_eff, W, M_matrix, xi_0, snr_t))
            rate[ss] = r.item()
            crlb[ss] = float(np.exp(-c.item()))
            obj[ss] = OMEGA * r.item() + c.item()
    return obj, rate, crlb

def main():
    torch.manual_seed(3407)

    _, H_test0 = get_data_tensor(data_source)
    H_test = H_test0[:, :test_size, :, :]                     # (K, B, M, Nt)
    B_test = H_test.shape[1]
    print(f"H_test (K, B, M, Nt): {tuple(H_test.shape)}")

    M_matrix = (A_dot.conj().T @ R_N_inv @ A_dot).to(H_test.device)  # (Nt, Nt)

    H_joint = to_joint_channel(H_test).to(device)             # (B, Nt, M)
    psi0 = torch.full((B_test,), desired_angle_rad_torch, device=device)

    # -- Joint models (two S_0 initialisation schemes + decay variant) -------
    obj_sel, rate_sel, crlb_sel = evaluate_joint('selection', H_joint, psi0, M_matrix, snr_dB_list, B_test)
    obj_fix, rate_fix, crlb_fix = evaluate_joint('fixed', H_joint, psi0, M_matrix, snr_dB_list, B_test)
    obj_decay, rate_decay, crlb_decay = evaluate_joint('fixed', H_joint, psi0, M_matrix, snr_dB_list, B_test, decay=True)

    # -- Baselines (frozen UPGA beamformer + connectivity masks) -------------
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

    obj_full = np.zeros(len(snr_dB_list))   # Full-connected
    rate_full = np.zeros(len(snr_dB_list))
    crlb_full = np.zeros(len(snr_dB_list))
    obj_sub  = np.zeros(len(snr_dB_list))   # Fixed sub-connected
    rate_sub = np.zeros(len(snr_dB_list))
    crlb_sub = np.zeros(len(snr_dB_list))
    obj_adp  = np.zeros(len(snr_dB_list))   # Adaptive (SelectionNet)
    rate_adp = np.zeros(len(snr_dB_list))
    crlb_adp = np.zeros(len(snr_dB_list))

    for ss, snr_dB in enumerate(snr_dB_list):
        snr_ss = 10 ** (snr_dB / 10)
        print(f'Evaluating baselines at SNR = {snr_dB} dB ...')

        with torch.no_grad():
            _, _, F_up, W_up, _, _ = upga.execute_PGA(
                H_test, xi_0, A_dot, R_N_inv,
                torch.tensor(snr_ss, dtype=torch.float32, device=device),
                n_iter_outer, n_iter_inner_J5, track_metrics=False)

            # Full-connected: use the UPGA's own optimized W.
            o_f, r_f, c_f = baseline_metrics(F_up, W_up, snr_ss)

            # Sub-connected: re-derive a matched W for the masked array.
            W_sub = compute_digital_precoder(H_test, F_up * fixed_mask) if REDERIVE_DIGITAL_W else W_up
            o_s, r_s, c_s = baseline_metrics(F_up * fixed_mask, W_sub, snr_ss)

            W_adp = compute_digital_precoder(H_test, F_up * S_hard.unsqueeze(0)) if REDERIVE_DIGITAL_W else W_up
            o_a, r_a, c_a = baseline_metrics(F_up * S_hard.unsqueeze(0), W_adp, snr_ss)

            obj_full[ss], rate_full[ss], crlb_full[ss] = o_f.item(), r_f.item(), float(np.exp(-c_f.item()))
            obj_sub[ss], rate_sub[ss], crlb_sub[ss] = o_s.item(), r_s.item(), float(np.exp(-c_s.item()))
            obj_adp[ss], rate_adp[ss], crlb_adp[ss] = o_a.item(), r_a.item(), float(np.exp(-c_a.item()))

    # ---- Plot helpers -------------------------------------------------------
    cmap = {
        'JointUPGANet (selection init)':  ('-o', 'red'),
        'JointUPGANet (fixed init)':      ('-^', 'purple'),
        'JointUPGANet + decay':           ('-v', 'orange'),
        'Full-connected':                 ('-d', 'black'),
        'Fixed sub-connected':            ('--s', 'blue'),
        'Adaptive (SelectionNet)':        ('-.v', 'green'),
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
        'JointUPGANet (selection init)': obj_sel,
        'JointUPGANet (fixed init)': obj_fix,
        'JointUPGANet + decay': obj_decay,
        'Full-connected': obj_full,
        'Fixed sub-connected': obj_sub,
        'Adaptive (SelectionNet)': obj_adp,
    }, r'$J = \omega R + \log(\mathrm{CRLB}^{-1})$',
        'JointUPGANet_obj_vs_SNR_{Nt}_{OMEGA}.png')

    # ---- Rate vs SNR
    plot_curves(snr_dB_list, {
        'JointUPGANet (selection init)': rate_sel,
        'JointUPGANet (fixed init)': rate_fix,
        'JointUPGANet + decay': rate_decay,
        'Full-connected': rate_full,
        'Fixed sub-connected': rate_sub,
        'Adaptive (SelectionNet)': rate_adp,
    }, r'$R$ [bits/s/Hz]', 'JointUPGANet_rate_vs_SNR_{Nt}_{OMEGA}.png')

    # ---- CRLB vs SNR
    plot_curves(snr_dB_list, {
        'JointUPGANet (selection init)': crlb_sel,
        'JointUPGANet (fixed init)': crlb_fix,
        'JointUPGANet + decay': crlb_decay,
        'Full-connected': crlb_full,
        'Fixed sub-connected': crlb_sub,
        'Adaptive (SelectionNet)': crlb_adp,
    }, r'$\mathrm{CRLB}$', 'JointUPGANet_CRB_vs_SNR_{Nt}_{OMEGA}.png')

    print(f'Saved figures to {directory_result}')


if __name__ == "__main__":
    main()
