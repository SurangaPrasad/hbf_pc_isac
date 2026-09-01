"""Train wideband UPGA models for each number of subcarriers.

For each K in ``K_list`` this script trains the three wideband variants
(Fixed-UPGA, Dynamic-UPGA/decay, GradReuse) on frequency-selective channels
synthesized from the narrowband dataset, and saves per-K checkpoints:

    UPGA_J5_K{K}.pth
    UPGA_decay_J5_K{K}.pth
    UPGA_GradReuse_K{K}.pth

The narrowband (K=1) checkpoints keep their legacy names, so existing scripts
are unaffected.

Run:  python main_train_subcarriers.py            # train all K in K_list
      python main_train_subcarriers.py 8 32       # train only these K values
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless: don't block on plt.show()
import matplotlib.pyplot as plt
import torch

from system_config import *
from utility import (
    get_data_tensor, synthesize_wideband_channels,
    build_sensing_matrices_per_subcarrier,
)
from PGA_models import (
    PGA_Unfold_JX, PGA_Unfold_JX_decay, PGA_Unfold_J_GradReuse,
    get_sum_loss_wideband,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TRAIN_LR = learning_rate
TRAIN_GRAD_CLIP_MAX_NORM = 1.0

# Subcarrier counts to train (overridable via CLI args).
K_list = [1, 8, 16, 24, 32, 40, 48, 56, 64]
if len(sys.argv) > 1:
    K_list = [int(k) for k in sys.argv[1:]]

# Which variants to train per K.
train_fixed = 1
train_decay = 1
train_gradreuse = 1

# Number of training epochs per K (n_epoch from system_config is for K=1;
# wideband batches are K times more expensive, so cap the default).
epochs_per_K = min(n_epoch, 30)


def wideband_ckpt_path(variant: str, K: int) -> str:
    """Checkpoint path for a wideband variant at a given K.

    K=1 keeps the legacy narrowband filenames so existing scripts keep working.
    """
    if K == 1:
        return {
            'fixed': model_file_name_UPGA_J5,
            'decay': model_file_name_UPGA_J5_decay,
            'gradreuse': model_file_name_UPGA_J_GradReuse,
        }[variant]
    return directory_model + f'UPGA_{variant}_J{n_iter_inner_J5}_K{K}.pth'


def make_batch(H_train, batch_size, K_wb, seed=None):
    """Sample one wideband training batch (channels + sensing matrices + SNR)."""
    if seed is not None:
        np.random.seed(seed)
    H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
    H_nb = torch.transpose(H_shuffled[:batch_size], 0, 1)          # (1, B, M, Nt)
    H_wb = synthesize_wideband_channels(H_nb, K_wb, n_taps=4, seed=None)
    cur_bs = H_nb.shape[1]
    snr_dB_train = np.random.permutation(
        np.tile(snr_dB_list, batch_size // len(snr_dB_list) + 1))[:cur_bs]
    snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)
    M_k, _ = build_sensing_matrices_per_subcarrier(K_wb)
    return H_wb, M_k, snr_train


def train_one_epoch(model, H_train, K_wb, optimizer, use_clip=False):
    """One training epoch over the wideband training set. Returns avg loss."""
    model.train()
    batch_losses = []
    n_batches = max(len(H_train[0]) // batch_size, 1)
    for i_batch in range(0, len(H_train[0]), batch_size):
        H_wb, M_k, snr_train = make_batch(H_train, batch_size, K_wb)

        if isinstance(model, PGA_Unfold_JX_decay):
            _, _, F, W, _ = model.execute_PGA_wideband(
                H_wb, M_k, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J5,
                track_metrics=False)
        else:
            _, _, F, W, _, _ = model.execute_PGA_wideband(
                H_wb, M_k, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J5,
                track_metrics=False)

        loss = get_sum_loss_wideband(F, W, H_wb, xi_0, M_k, snr_train)

        optimizer.zero_grad()
        loss.backward()
        if use_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP_MAX_NORM)
        optimizer.step()

        batch_losses.append(loss.item())
        print(f"    Batch [{i_batch//batch_size+1}/{n_batches}], Loss: {loss.item():.4f}")

    return sum(batch_losses) / len(batch_losses)


def plot_loss_curve(epoch_losses, title, path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def train_for_K(K_wb, H_train):
    """Train all enabled variants for one subcarrier count."""
    print(f"\n================ Training K = {K_wb} ================")
    torch.manual_seed(3407)

    # ---- Fixed-UPGA (wideband) ----
    if train_fixed:
        ckpt = wideband_ckpt_path('fixed', K_wb)
        if os.path.exists(ckpt):
            print(f"  [Fixed-UPGA] checkpoint exists, skipping: {ckpt}")
        else:
            print(f"  [Fixed-UPGA] training -> {ckpt}")
            model = PGA_Unfold_JX(step_size_UPGA_J5).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
            losses = []
            for ep in range(epochs_per_K):
                avg = train_one_epoch(model, H_train, K_wb, optimizer)
                losses.append(avg)
                print(f"    Epoch [{ep+1}/{epochs_per_K}], Average Loss: {avg:.4f}")
            torch.save(model.state_dict(), ckpt)
            plot_loss_curve(losses, f'Training Loss (Fixed-UPGA, K={K_wb})',
                            directory_model + f'UPGA_J{n_iter_inner_J5}_K{K_wb}_loss.png')

    # ---- Dynamic-UPGA / decay (wideband) ----
    if train_decay:
        ckpt = wideband_ckpt_path('decay', K_wb)
        if os.path.exists(ckpt):
            print(f"  [Dynamic-UPGA] checkpoint exists, skipping: {ckpt}")
        else:
            print(f"  [Dynamic-UPGA] training -> {ckpt}")
            model = PGA_Unfold_JX_decay(step_size_UPGA_J5_decay).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
            losses = []
            for ep in range(epochs_per_K):
                avg = train_one_epoch(model, H_train, K_wb, optimizer)
                losses.append(avg)
                print(f"    Epoch [{ep+1}/{epochs_per_K}], Average Loss: {avg:.4f}")
            torch.save(model.state_dict(), ckpt)
            plot_loss_curve(losses, f'Training Loss (Dynamic-UPGA, K={K_wb})',
                            directory_model + f'UPGA_decay_J{n_iter_inner_J5}_K{K_wb}_loss.png')

    # ---- GradReuse (wideband) ----
    if train_gradreuse:
        ckpt = wideband_ckpt_path('gradreuse', K_wb)
        if os.path.exists(ckpt):
            print(f"  [GradReuse] checkpoint exists, skipping: {ckpt}")
        else:
            print(f"  [GradReuse] training -> {ckpt}")
            model = PGA_Unfold_J_GradReuse(step_size_UPGA_J_GradReuse).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
            losses = []
            for ep in range(epochs_per_K):
                avg = train_one_epoch(model, H_train, K_wb, optimizer, use_clip=True)
                losses.append(avg)
                print(f"    Epoch [{ep+1}/{epochs_per_K}], Average Loss: {avg:.4f}  "
                      f"(F fallbacks: {model.grad_recalc_count})")
            torch.save(model.state_dict(), ckpt)
            plot_loss_curve(losses, f'Training Loss (GradReuse, K={K_wb})',
                            directory_model + f'UPGA_GradReuse_J{n_iter_inner_J5}_K{K_wb}_loss.png')


def main():
    H_train, _ = get_data_tensor(data_source)
    print(f'Training data: {tuple(H_train.shape)}')
    print(f'K list: {K_list}, epochs per K: {epochs_per_K}')

    for K_wb in K_list:
        train_for_K(K_wb, H_train)

    print('\nAll requested K values trained (or skipped where checkpoints exist).')


if __name__ == '__main__':
    main()
