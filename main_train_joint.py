"""Train the JointUPGANet (joint deep-unfolding of F, S and W).

Two initialisation schemes for the connection matrix S_0 are supported:

  * ``selection`` (default) — S_0 comes from SelectionNet (a Gumbel-softmax MLP
    that maps the channel and sensing direction to a row-stochastic mask).
  * ``fixed``            — S_0 is the fixed block sub-connected mask (every RF
    chain serves N_antennas/N_rf antennas); the unfolding then refines S the
    same way as the selection variant.

Only the per-layer step sizes (mu, kappa, lambda) and (for ``selection``) the
SelectionNet weights are learnable; F_0 and W_0 are re-initialised from the
channel for every batch.

Run:
    python main_train_joint.py              # selection variant
    python main_train_joint.py fixed        # fixed-mask variant
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from system_config import *
from utility import get_data_tensor, safe_legend
from joint_upganet import JointUPGANet, JointUPGANet_decay, get_joint_loss, initialize_joint

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Joint-model specific hyper-parameters -------------------------------
JOINT_LR = 5e-3              # Adam LR (step sizes + SelectionNet)
JOINT_TAU_START = 2.0        # Gumbel temperature at epoch 0
JOINT_TAU_END = 0.1          # Gumbel temperature at the last epoch
JOINT_HARD_FINAL = 5         # last N epochs with hard STE (matches eval)
JOINT_GRAD_CLIP = 1.0        # global grad-norm clip

# Option B: train with the HARD one-hot mask (straight-through estimator) for
# the WHOLE run, for BOTH variants, so training and evaluation use the same
# (hard) objective.  This removes the train/eval mismatch that made the hard
# eval graph worse than the soft one.  The Gumbel temperature is still annealed
# (it controls the quality of the STE gradient through the soft probabilities).
JOINT_HARD_ALL = True

# Reuse the legacy outer/inner-iteration schedule (see system_config.py).
N_OUTER = n_iter_outer        # I outer iterations
N_INNER = n_iter_inner_J5     # J inner steps per outer iteration


def model_filename(s_init: str, decay: bool = False) -> str:
    """Checkpoint path for a given S_0 initialisation scheme (+ decay variant)."""
    tag = "" if s_init == "selection" else f"_{s_init}"
    if decay:
        return directory_model + f'JointUPGANet{tag}_decay_I{N_OUTER}_J{N_INNER}.pth'
    return directory_model + f'JointUPGANet{tag}_I{N_OUTER}_J{N_INNER}.pth'


def anneal_tau(epoch, n_epoch, tau_start=JOINT_TAU_START, tau_end=JOINT_TAU_END):
    """Exponential Gumbel temperature decay over epochs."""
    return tau_start * (tau_end / tau_start) ** (epoch / max(1, n_epoch - 1))


def to_joint_channel(H_kb: torch.Tensor) -> torch.Tensor:
    """(K, B, M, Nt) -> (B, Nt, M) using the single frequency band."""
    return H_kb[0].transpose(1, 2)   # strip K, swap users<->antennas


def main(s_init: str = "fixed", decay: bool = False):
    assert s_init in ("selection", "fixed")
    torch.manual_seed(3407)

    H_train, _ = get_data_tensor(data_source)
    print(f"H_train (K, B, M, Nt): {tuple(H_train.shape)}  "
          f"(s_init={s_init}, decay={decay})")

    # ---- Sensing Fisher-like matrix in antenna space (shared across batch).
    M_matrix = (A_dot.conj().T @ R_N_inv @ A_dot).to(H_train.device)   # (Nt, Nt)

    if decay:
        model = JointUPGANet_decay(step_size=step_size_joint_decay, n_antennas=Nt,
                                   n_rf_chains=Nrf, n_users=M, s_init=s_init).to(device)
    else:
        model = JointUPGANet(step_size=step_size_joint, n_antennas=Nt,
                             n_rf_chains=Nrf, n_users=M, s_init=s_init).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=JOINT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epoch_losses = []

    for i_epoch in range(n_epoch):
        tau = anneal_tau(i_epoch, n_epoch)
        # Option B: train with a HARD one-hot S (via the straight-through
        # estimator) from epoch 0 for BOTH variants, so the training objective
        # matches the hard evaluation protocol.  This avoids the abrupt
        # soft->hard loss jump and the train/eval mismatch that made the hard
        # eval graph worse than the soft one.  The Gumbel temperature is still
        # annealed because it controls the STE gradient quality through the
        # soft probabilities.
        hard_mode = JOINT_HARD_ALL
        if hard_mode:
            tau = JOINT_TAU_END

        batch_losses = []

        # Shuffle along the batch axis (same pattern as main_train.py).
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H_batch = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H_batch.shape[1]

            # Balanced per-SNR draw, as in main_train.py.
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)

            # Channel in (B, Nt, M) layout + fixed sensing direction.
            H_joint = to_joint_channel(H_batch).to(device)          # (B, Nt, M)
            psi0 = torch.full((cur_bs,), desired_angle_rad_torch, device=device)

            # tau/hard only affect the 'selection' variant (ignored for 'fixed').
            _, _, F, S, W = model.execute_PGA(
                H_joint, psi0, M_matrix, OMEGA, snr_train,
                N_OUTER, N_INNER, xi_0, tau=tau, hard=hard_mode, track_metrics=False)

            loss = get_joint_loss(F, S, W, H_joint, M_matrix, OMEGA, xi_0, snr_train)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), JOINT_GRAD_CLIP)
            optimizer.step()

            batch_losses.append(loss.item())
            print(f"Epoch [{i_epoch+1}/{n_epoch}] "
                  f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}] "
                  f"Loss: {loss.item():.4f}, tau: {tau:.3f}")

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step()
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    # ---- Save model + loss curve
    torch.save(model.state_dict(), model_filename(s_init, decay))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    title_tag = ' + decay' if decay else ''
    plt.title(f'JointUPGANet{title_tag} ({s_init}) Training Loss (I={N_OUTER}, J={N_INNER})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_tag = f'{s_init}_decay' if decay else s_init
    plt.savefig(directory_result + f'JointUPGANet_{loss_tag}_loss_I{N_OUTER}_J{N_INNER}.png', dpi=300)
    print(f"Saved loss curve to {directory_result}")


if __name__ == "__main__":
    # Usage:
    #   python main_train_joint.py [selection|fixed] [decay]
    # e.g. python main_train_joint.py fixed decay
    s_init = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    decay = len(sys.argv) > 2 and sys.argv[2].lower() == "decay"
    if s_init not in ("selection", "fixed"):
        raise SystemExit(f"usage: python main_train_joint.py [selection|fixed] [decay], got {s_init!r}")
    main(s_init, decay)
