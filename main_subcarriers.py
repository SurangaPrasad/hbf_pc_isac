"""Objective vs number of subcarriers (wideband OFDM overlapping allocation).

Loads the K=1 trained step sizes (UPGA_J5.pth / UPGA_J5_decay.pth) and runs
the wideband variants of PGA_Unfold_JX, PGA_Unfold_JX_decay and
PGA_Unfold_J_GradReuse for each subcarrier count in ``K_list``.  The analog
precoder F is frequency-flat (pooled gradient across subcarriers); the digital
precoders W[k] are per-subcarrier.  The final objective

    g = OMEGA * R + log(CRLB^-1)

is averaged over the last ``avg_last`` outer iterations and plotted against
the number of subcarriers.

Run:  python main_subcarriers.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from system_config import *
from utility import (
    get_data_tensor, synthesize_wideband_channels,
    build_sensing_matrices_per_subcarrier, safe_legend,
)
from PGA_models import PGA_Unfold_JX, PGA_Unfold_JX_decay, PGA_Unfold_J_GradReuse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------- sweep configuration -----------------------------
K_list = [1, 8, 16, 24, 32, 40, 48, 56, 64]   # subcarrier counts to evaluate
avg_last = 10                                  # average objective over last I iters
seed = 3407                                    # channel synthesis seed

run_JX = 1
run_decay = 1
run_gradreuse = 1

# If 1, use the per-K trained checkpoints (from main_train_subcarriers.py)
# when they exist; otherwise fall back to the K=1 trained step sizes.
use_per_K_checkpoints = 1


def wideband_ckpt_path(variant: str, K_wb: int) -> str:
    """Checkpoint path for a wideband variant at a given K.

    K=1 maps to the legacy narrowband filenames; K>1 uses the per-K names
    written by main_train_subcarriers.py.
    """
    if K_wb == 1:
        return {
            'fixed': model_file_name_UPGA_J5,
            'decay': model_file_name_UPGA_J5_decay,
            'gradreuse': model_file_name_UPGA_J_GradReuse,
        }[variant]
    return directory_model + f'UPGA_{variant}_J{n_iter_inner_J5}_K{K_wb}.pth'


def load_variant_checkpoint(model, variant: str, K_wb: int):
    """Load the best available checkpoint for a variant at a given K.

    Preference: per-K trained checkpoint (if enabled and it exists) ->
    K=1 trained checkpoint -> keep the model's default step sizes.
    Returns the path actually loaded (or None).
    """
    candidates = []
    if use_per_K_checkpoints:
        candidates.append(wideband_ckpt_path(variant, K_wb))
    candidates.append(wideband_ckpt_path(variant, 1))

    for path in candidates:
        if os.path.exists(path):
            state = torch.load(path, map_location=device)
            if 'step_size' in state and state['step_size'].shape != model.step_size.shape:
                print(f'  [{variant}] checkpoint step_size {tuple(state["step_size"].shape)} '
                      f'!= model {tuple(model.step_size.shape)}; keeping model step sizes.')
                state = {k: v for k, v in state.items() if k != 'step_size'}
            model.load_state_dict(state, strict=False)
            return path
    print(f'  [{variant}] no checkpoint found; using default step sizes.')
    return None

# ----------------------------- result cache -----------------------------
import scipy.io

def get_cache_file_name():
    return directory_result + 'subcarrier_sweep_cache_' + str(Nt) + '_' + str(OMEGA) + '.mat'

def save_cache(file_path, namespace):
    payload = {'K_list': np.asarray(K_list, dtype=float)}
    for key in ('obj_JX', 'obj_decay', 'obj_gradreuse'):
        if key in namespace and namespace[key] is not None:
            payload[key] = np.asarray(namespace[key])
    # Store decay inner-iteration histories as a padded 2-D array.
    if namespace.get('inner_iter_histories'):
        max_len = max(len(h) for h in namespace['inner_iter_histories'])
        padded = np.full((len(namespace['inner_iter_histories']), max_len), -1, dtype=np.int32)
        for i, h in enumerate(namespace['inner_iter_histories']):
            padded[i, :len(h)] = h
        payload['inner_iter_histories'] = padded
    scipy.io.savemat(file_path, payload)
    print(f'Saved subcarrier sweep cache to {file_path}')

def load_cache(file_path):
    cache = scipy.io.loadmat(file_path, allow_pickle=True, simplify_cells=True)
    out = {}
    for key in ('obj_JX', 'obj_decay', 'obj_gradreuse', 'inner_iter_histories'):
        if key in cache:
            out[key] = cache[key]
    return out


def run_wideband_model(model, H_wb, M_k, snr_t, n_outer, n_inner):
    """Run a wideband execute_PGA and return the final averaged objective."""
    with torch.no_grad():
        rates, crb_fes, F, W, *_ = model.execute_PGA_wideband(
            H_wb, M_k, R_N_inv, snr_t, n_outer, n_inner, track_metrics=True)
    obj_curve = (OMEGA * rates.mean(dim=1) + crb_fes.mean(dim=1)).cpu().numpy()
    return obj_curve


def main():
    torch.manual_seed(seed)

    # ---------------- Load narrowband reference channels ----------------
    H_train, H_test0 = get_data_tensor(data_source)
    H_ref = H_train[:, :test_size, :, :]           # (1, B, M, Nt)
    B = H_ref.shape[1]
    print(f'Reference narrowband channels: {tuple(H_ref.shape)}')

    # Per-sample SNR tensor (same protocol as training).
    snr_t = torch.full((B,), snr, dtype=torch.float32, device=device)

    obj_JX = []
    obj_decay = []
    obj_gradreuse = []
    inner_iter_histories = []

    for K_wb in K_list:
        print(f'\n================ K = {K_wb} subcarriers ================')

        # ---- Synthesize wideband channels (center subcarrier = reference) ----
        H_wb = synthesize_wideband_channels(H_ref, K_wb, n_taps=4, delay_spread=3.0, seed=seed)
        print(f'  Wideband channels: {tuple(H_wb.shape)}')

        # ---- Per-subcarrier sensing matrices M[k] ----
        M_k, _ = build_sensing_matrices_per_subcarrier(K_wb)
        M_k = M_k.to(device)
        print(f'  Sensing matrices M[k]: {tuple(M_k.shape)}')

        # ---- Fixed-J UPGA (wideband) ----
        if run_JX:
            model = PGA_Unfold_JX(step_size_UPGA_J5).to(device)
            ckpt = load_variant_checkpoint(model, 'fixed', K_wb)
            model.eval()
            curve = run_wideband_model(model, H_wb, M_k, snr_t, n_iter_outer, n_iter_inner_J5)
            obj_JX.append(curve[-avg_last:].mean())
            print(f'  [Fixed-UPGA]      ckpt={os.path.basename(ckpt) if ckpt else "default"}  '
                  f'final objective = {obj_JX[-1]:.4f}')

        # ---- Dynamic-UPGA / decay (wideband) ----
        if run_decay:
            model_decay = PGA_Unfold_JX_decay(step_size_UPGA_J5_decay).to(device)
            ckpt = load_variant_checkpoint(model_decay, 'decay', K_wb)
            model_decay.eval()
            curve_d = run_wideband_model(model_decay, H_wb, M_k, snr_t, n_iter_outer, n_iter_inner_J5)
            obj_decay.append(curve_d[-avg_last:].mean())
            inner_iter_histories.append(list(model_decay.inner_iter_history))
            print(f'  [Dynamic-UPGA]    ckpt={os.path.basename(ckpt) if ckpt else "default"}  '
                  f'final objective = {obj_decay[-1]:.4f}  '
                  f'(inner iters used: {model_decay.inner_iter_history[:5]}...)')

        # ---- GradReuse (wideband) ----
        if run_gradreuse:
            model_gr = PGA_Unfold_J_GradReuse(step_size_UPGA_J_GradReuse).to(device)
            ckpt = load_variant_checkpoint(model_gr, 'gradreuse', K_wb)
            model_gr.eval()
            curve_g = run_wideband_model(model_gr, H_wb, M_k, snr_t, n_iter_outer, n_iter_inner_J5)
            obj_gradreuse.append(curve_g[-avg_last:].mean())
            print(f'  [GradReuse]       ckpt={os.path.basename(ckpt) if ckpt else "default"}  '
                  f'final objective = {obj_gradreuse[-1]:.4f}  '
                  f'(F fallbacks: {model_gr.grad_recalc_count})')

    # ---------------- Plot objective vs number of subcarriers ----------------
    plt.figure(figsize=(8, 5.2))
    if run_JX:
        plt.plot(K_list, obj_JX, '--o', color='red', linewidth=3, markersize=8,
                 label=label_UPGA_J5)
    if run_decay:
        plt.plot(K_list, obj_decay, '-d', color='green', linewidth=3, markersize=8,
                 label=label_UPGA_J5_decay)
    if run_gradreuse:
        plt.plot(K_list, obj_gradreuse, '-x', color='magenta', linewidth=3, markersize=8,
                 label=label_UPGA_J_GradReuse)

    plt.xlabel(r'Number of subcarriers $(K)$', fontsize=14)
    plt.ylabel(r'$\omega R + \log(\mathrm{CRLB}^{-1})$', fontsize=14)
    plt.title(rf'$N={Nt}, M={M}, N_{{\mathrm{{RF}}}}={Nrf}, \mathrm{{SNR}}={snr_dB}\,\mathrm{{dB}}, \omega={OMEGA}$',
              fontsize=13)
    plt.grid()
    safe_legend(loc='best', fontsize=11, frameon=False)
    plt.savefig(directory_result + 'objective_vs_subcarriers_' + str(Nt) + '_' + str(OMEGA) + '.png',
                bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'objective_vs_subcarriers_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
    print(f'\nSaved figure to {directory_result}objective_vs_subcarriers_{Nt}_{OMEGA}.png')

    # ---------------- Save cache ----------------
    save_cache(get_cache_file_name(), dict(
        obj_JX=obj_JX if run_JX else None,
        obj_decay=obj_decay if run_decay else None,
        obj_gradreuse=obj_gradreuse if run_gradreuse else None,
        inner_iter_histories=inner_iter_histories if run_decay else None,
    ))


if __name__ == '__main__':
    main()
