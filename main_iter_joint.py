"""Plot objective vs outer iterations at SNR = 12 dB, trained and untrained.

Six curves comparing three methods * (trained / untrained), where "trained"
means loading the respective ``state_dict`` and "untrained" means default step
sizes:

  1. JointUPGANet (fixed init)   - trained
  2. JointUPGANet (fixed init)   - untrained
  3. Fixed sub-connected         - trained
  4. Fixed sub-connected         - untrained
  5. Full-connected              - trained
  6. Full-connected              - untrained

The physics objective is ``J = omega * R + log(CRLB^-1)``, evaluated after every
outer iteration at a fixed SNR of 12 dB.

Run:  python main_iter_joint.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from system_config import *
from utility import get_data_tensor, safe_legend
from PGA_models import PGA_Unfold_JX, PGA_Unfold_JX_partial
from joint_upganet import (
    JointUPGANet, build_fixed_subconnected_mask,
    get_sum_rate_joint, get_crb_joint, initialize_joint, load_joint_state_dict,
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


def default_upga_step_size() -> torch.Tensor:
    """Untrained UPGA step sizes (all set to ``step_size_fixed``)."""
    return torch.full([N_INNER, N_OUTER, K + 1], step_size_fixed, device=device)


def unroll_joint(H_joint, psi0, M_matrix, snr_t, trained=True):
    """JointUPGANet (fixed init); optionally loads the trained state_dict."""
    model = JointUPGANet(n_outer=N_OUTER, n_inner=N_INNER, n_antennas=Nt, n_rf_chains=Nrf, n_users=M, s_init='fixed', step_size=step_size_joint).to(device)
    if trained:
        load_joint_state_dict(model, torch.load(joint_model_path('fixed'), map_location=device), N_INNER)
    model.eval()

    obj_iter = np.zeros(N_OUTER)
    with torch.no_grad():
        F0, W0 = initialize_joint(H_joint, snr_t, Nrf)
        S = model.fixed_s0.expand(H_joint.shape[0], -1, -1).clone()
        F, W = F0, W0
        for ii, layer in enumerate(model.layers):
            F, S, W = layer(F, S, W, H_joint, psi0, M_matrix, OMEGA, snr_t)
            # Report the objective with a HARD one-hot sub-connected mask
            # (one RF chain per antenna), matching the eval protocol. The soft
            # S is kept for the next layer (the network is trained with soft S);
            # only the tracked objective uses the hardened mask.
            S_hard = torch.zeros_like(S)
            S_hard.scatter_(-1, S.argmax(dim=-1).unsqueeze(-1), 1.0)
            F_eff = F * S_hard
            r = get_sum_rate_joint(H_joint, F_eff, W, snr_t)
            c = torch.mean(get_crb_joint(F_eff, W, M_matrix, xi_0, snr_t))
            obj_iter[ii] = OMEGA * r.item() + c.item()
    return obj_iter


def unroll_upga(H_test, mask=None, trained=True):
    """Run an UPGA (full or fixed sub-connected).

    When ``mask`` is None the full-connected UPGA is used; otherwise the
    partial (sub-connected) UPGA is instantiated with the fixed block mask.
    If ``trained`` is True the corresponding state_dict is loaded (step sizes
    only; the mask buffer of the partial model is *not* overwritten).
    """
    step_size = default_upga_step_size()
    if mask is None:
        model = PGA_Unfold_JX(step_size).to(device)
        model_path = model_file_name_UPGA_J5
    else:
        model = PGA_Unfold_JX_partial(step_size, mask=mask).to(device)
        model_path = model_file_name_UPGA_J5

    if trained:
        state = torch.load(model_path, map_location=device)
        state = {k: v for k, v in state.items() if k not in ('mask',)}
        model.load_state_dict(state, strict=False)

    model.eval()

    with torch.no_grad():
        rates, crb_fes, F, W, _, _ = model.execute_PGA(
            H_test, xi_0, A_dot, R_N_inv, snr,
            N_OUTER, N_INNER, track_metrics=True)

    obj = OMEGA * rates.mean(dim=1).cpu().numpy() + crb_fes.mean(dim=1).cpu().numpy()
    return obj


def main():
    torch.manual_seed(3407)

    H_Train0, H_test0 = get_data_tensor(data_source)
    H_test = H_Train0[:, :test_size, :, :]                     # (K, B, M, Nt)
    B_test = H_test.shape[1]
    print(f'H_test (K, B, M, Nt): {tuple(H_test.shape)}')

    M_matrix = (A_dot.conj().T @ R_N_inv @ A_dot).to(H_test.device)  # (Nt, Nt)

    H_joint = to_joint_channel(H_test).to(device)             # (B, Nt, M)
    psi0 = torch.full((B_test,), desired_angle_rad_torch, device=device)
    snr_t = torch.full((B_test,), snr, dtype=torch.float32, device=device)

    block_mask = build_fixed_subconnected_mask(Nt, Nrf).to(device)   # (Nt, Nrf)

    print('JointUPGANet (fixed init), trained ...')
    obj_joint_tr = unroll_joint(H_joint, psi0, M_matrix, snr_t, trained=True)
    print('JointUPGANet (fixed init), untrained ...')
    obj_joint_un = unroll_joint(H_joint, psi0, M_matrix, snr_t, trained=False)

    print('Fixed sub-connected, trained ...')
    obj_sub_tr = unroll_upga(H_test, mask=block_mask, trained=True)
    print('Fixed sub-connected, untrained ...')
    obj_sub_un = unroll_upga(H_test, mask=block_mask, trained=False)

    print('Full-connected, trained ...')
    obj_full_tr = unroll_upga(H_test, mask=None, trained=True)
    print('Full-connected, untrained ...')
    obj_full_un = unroll_upga(H_test, mask=None, trained=False)

    iter_x = np.arange(1, N_OUTER + 1)

    # ---- Objective vs outer iterations ------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(iter_x, obj_joint_tr, '--^', color='red', linewidth=3, markersize=6, markevery=5, label='JointUPGANet (fixed init), trained')
    plt.plot(iter_x, obj_joint_un, '--^', color='red', linewidth=3, markersize=6, markevery=5, label='JointUPGANet (fixed init), untrained')
    plt.plot(iter_x, obj_sub_tr, '--s', color='blue', linewidth=3, markersize=6, markevery=5, label='Fixed sub-connected, trained')
    plt.plot(iter_x, obj_sub_un, '--s', color='blue', linewidth=3, markersize=6, markevery=5, label='Fixed sub-connected, untrained')
    plt.plot(iter_x, obj_full_tr, '--d', color='green', linewidth=3, markersize=6, markevery=5, label='Full-connected, trained')
    plt.plot(iter_x, obj_full_un, '--d', color='green', linewidth=3, markersize=6, markevery=5, label='Full-connected, untrained')


    plt.xlabel(r'Number of outer iterations $(I)$', fontsize=14)
    plt.ylabel(r'$\omega R + \log(\mathrm{CRLB}^{-1})$', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=10, labelspacing=0.10)
    plt.tight_layout()
    plt.savefig(directory_result + f'objective_vs_iter_{Nt}_{OMEGA}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)

    for label, obj in [('JointUPGANet (fixed init), trained', obj_joint_tr),
                       ('JointUPGANet (fixed init), untrained', obj_joint_un),
                       ('Fixed sub-connected, trained', obj_sub_tr),
                       ('Fixed sub-connected, untrained', obj_sub_un),
                       ('Full-connected, trained', obj_full_tr),
                       ('Full-connected, untrained', obj_full_un)]:
        print(f'  {label:45s} J={obj[-1]:.4f}')
    print(f'Saved figure to {directory_result}')


if __name__ == "__main__":
    main()