import os
import numpy as np
import matplotlib.pyplot as plt
import inspect
import torch
from utility import *
from PGA_models import PGA_Unfold_JX, get_sum_loss
from SelectionNet import SelectionNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TRAIN_LR = learning_rate
TRAIN_SCHEDULER_FACTOR = 0.1
TRAIN_SCHEDULER_PATIENCE = 3
TRAIN_GRAD_CLIP_MAX_NORM = 1.0

# SelectionNet-specific training hyper-parameters
SELNET_LR = 1e-3                    # Adam LR for the assignment network
SELNET_TAU_START = 2.0              # Gumbel temperature at epoch 0 (high = explore)
SELNET_TAU_END = 0.1                # Gumbel temperature at the last epoch (low = sharpen)
SELNET_LOAD_BALANCE_WEIGHT = 0.0    # optional column_load regularizer; 0 = off for now


def build_optimizer_and_scheduler(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    scheduler_kwargs = {
        'mode': 'min',
        'factor': TRAIN_SCHEDULER_FACTOR,
        'patience': TRAIN_SCHEDULER_PATIENCE,
    }
    if 'verbose' in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__).parameters:
        scheduler_kwargs['verbose'] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    return optimizer, scheduler


def clip_gradients(model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP_MAX_NORM)


def anneal_tau(epoch, n_epoch, tau_start=SELNET_TAU_START, tau_end=SELNET_TAU_END):
    """Exponential temperature decay: high tau explores early, low tau sharpens
    S toward one-hot at the end (standard for Gumbel-softmax training)."""
    return tau_start * (tau_end / tau_start) ** (epoch / max(1, n_epoch - 1))


def load_pretrained_upga(model_path, n_inner, device):
    """Rebuild the frozen UPGA beamformer that produces F, W.

    Must instantiate the exact class (PGA_Unfold_JX) with the exact step_size
    shape used during its own training (J, n_iter_outer, K+1), then load the
    state_dict. requires_grad=False keeps it frozen so only SelectionNet trains.
    """
    step_size = torch.full(
        [n_inner, n_iter_outer, K + 1], step_size_fixed,
        device=device, requires_grad=False,
    )
    model = PGA_Unfold_JX(step_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_selectionnet():
    """Train SelectionNet (learnable antenna->RF-chain assignment).

    The sub-connected mask is learned per sample: S = SelectionNet(H, psi0).
    A frozen pre-trained UPGA provides F, W under no_grad; we then gate the
    analog precoder F_eff = F * S and backprop the unsupervised physics loss
    (get_sum_loss) through S into the SelectionNet parameters only.
    """
    torch.manual_seed(3407)

    # ---- Data: H_train is (K_freq, B, M, Nt); SelectionNet wants (B, Nt, M)
    H_train, _ = get_data_tensor(data_source)
    print(f"H_train shape (K, B, M, Nt): {tuple(H_train.shape)}")

    # ---- Frozen beamformer + trainable assignment network
    upga = load_pretrained_upga(model_file_name_UPGA_J5, n_iter_inner_J5, device)
    selnet = SelectionNet(n_antennas=Nt, n_rf_chains=Nrf, n_users=M).to(device)
    optimizer = torch.optim.Adam(selnet.parameters(), lr=SELNET_LR)

    epoch_losses = []

    for i_epoch in range(n_epoch):
        batch_losses = []
        tau = anneal_tau(i_epoch, n_epoch)

        # Shuffle along the batch axis, same pattern as main_train.py
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            # (K, B, M, Nt) -> (B, Nt, M): strip the single frequency band and
            # swap M<->Nt so channels are (antennas, users) for SelectionNet.
            H_batch = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H_batch.shape[1]

            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)

            H_sel = H_batch[0].transpose(1, 2)                     # (B, Nt, M) complex
            psi0 = torch.full((cur_bs,), desired_angle_rad_torch, device=device)

            # ---- SelectionNet forward: soft S for training
            S, logits = selnet(H_sel, psi0, tau=tau, hard=False)   # (B, Nt, Nrf)

            # ---- Frozen UPGA gives the beamformer under no_grad
            with torch.no_grad():
                _, _, F, W, _, _ = upga.execute_PGA(
                    H_batch, xi_0, A_dot, R_N_inv, snr_train,
                    n_iter_outer, n_iter_inner_J5, track_metrics=False)
                # F: (K, B, Nt, Nrf), W: (K, B, Nrf, M) — keep the K dim, the
                # loss functions (get_sum_rate / get_crb_fe) expect 4D tensors.

            # ---- Apply the learnable sub-connected mask, then physics loss.
            # S is (B, Nt, Nrf); unsqueeze(0) -> (1, B, Nt, Nrf) to broadcast
            # against the K=1 dimension of F.
            F_eff = F * S.unsqueeze(0)
            loss = get_sum_loss(F_eff, W, H_batch, xi_0, A_dot, R_N_inv, snr_train)

            # ---- Optional load-balancing regularizer (off by default)
            if SELNET_LOAD_BALANCE_WEIGHT > 0:
                loads = selnet.column_load(S)                      # (B, Nrf)
                load_penalty = loads.std(dim=-1).mean()
                loss = loss + SELNET_LOAD_BALANCE_WEIGHT * load_penalty

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(selnet)
            optimizer.step()

            batch_losses.append(loss.item())
            print(f"Batch [{i_batch // batch_size + 1}/{len(H_train[0]) // batch_size}], "
                  f"Loss: {loss.item():.4f}, tau: {tau:.3f}")

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch + 1}/{n_epoch}], Average Loss: {avg_loss:.4f}, tau: {tau:.3f}")

    # ---- Save
    torch.save(selnet.state_dict(), directory_model + f'SelectionNet_J{n_iter_inner_J5}.pth')

    # ---- Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('SelectionNet Training Loss over Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(directory_model + 'SelectionNet_loss.png', dpi=300)
    plt.show()

    # ---- Sanity: hard-mode assignment coverage (antennas per RF chain).
    # Rebuild a small batch from the training set since the loop's H_sel is gone.
    with torch.no_grad():
        H_chk = H_train[:, :batch_size].transpose(1, 0)            # (B, K, M, Nt)
        H_chk = H_chk.transpose(0, 1)[:, :, :, :]                  # (K, B, M, Nt)
        H_sel_chk = H_chk[0].transpose(1, 2)                       # (B, Nt, M)
        psi0_chk = torch.full((batch_size,), desired_angle_rad_torch, device=device)
        S_hard, _ = selnet(H_sel_chk, psi0_chk, tau=0.05, hard=True)
        loads = selnet.column_load(S_hard).cpu()
        print(f"Hard assignment antenna load per RF chain: {loads[0].tolist()} "
              f"(sum = {loads[0].sum().item():.0f} antennas, expected {Nt})")

    # ---- Post-training evaluation: objective vs SNR for the three connection schemes.
    plot_selectionnet_objective_vs_snr()


def plot_selectionnet_objective_vs_snr():
    """Sweep SNR and plot the physics objective J = OMEGA*R + mean(log CRLB) for:

      1. the trained SelectionNet (per-sample learned sub-connected mask),
      2. a fixed hand-built sub-connected mask,
      3. the full-connected version (no mask).

    All three share the same frozen UPGA J5 beamformer (F, W), so the only
    difference is the antenna->RF-chain mask applied to F — exactly the same
    protocol used during SelectionNet training.
    """
    torch.manual_seed(3407)

    # ---- Test data: (K, B_test, M, Nt)
    _, H_test0 = get_data_tensor(data_source)
    H_test = H_test0[:, :test_size, :, :]
    B_test = H_test.shape[1]

    # ---- Frozen beamformer + trained SelectionNet
    upga = load_pretrained_upga(model_file_name_UPGA_J5, n_iter_inner_J5, device)
    selnet = SelectionNet(n_antennas=Nt, n_rf_chains=Nrf, n_users=M).to(device)
    selnet.load_state_dict(torch.load(
        directory_model + f'SelectionNet_J{n_iter_inner_J5}.pth', map_location=device))
    selnet.eval()

    # ---- Fixed sub-connected mask (uniform block: Nt/Nrf antennas per RF chain).
    fixed_mask = generage_partial_connection_mask(Nt, Nrf).real.to(device)  # (Nt, Nrf)

    # ---- SelectionNet mask: depends only on H (and psi0), so compute once.
    H_sel = H_test[0].transpose(1, 2)                               # (B, Nt, M)
    psi0 = torch.full((B_test,), desired_angle_rad_torch, device=device)
    with torch.no_grad():
        S_hard, _ = selnet(H_sel, psi0, tau=0.05, hard=True)        # (B, Nt, Nrf)

    obj_selnet = np.zeros(len(snr_dB_list))
    obj_sub = np.zeros(len(snr_dB_list))
    obj_full = np.zeros(len(snr_dB_list))

    for ss, snr_dB in enumerate(snr_dB_list):
        snr_ss = 10 ** (snr_dB / 10)
        print(f'Evaluating SelectionNet objective at SNR = {snr_dB} dB ...')

        # Frozen UPGA gives F (K, B, Nt, Nrf), W (K, B, Nrf, M) under no_grad.
        with torch.no_grad():
            _, _, F, W, _, _ = upga.execute_PGA(
                H_test, xi_0, A_dot, R_N_inv,
                torch.tensor(snr_ss, dtype=torch.float32, device=device),
                n_iter_outer, n_iter_inner_J5, track_metrics=False)

            def objective(F_eff, W_eff):
                rate = get_sum_rate(H_test, F_eff, W_eff, snr_ss)
                crb = torch.mean(get_crb_fe(H_test, F_eff, W_eff, xi_0, A_dot, R_N_inv, snr_ss))
                return OMEGA * rate + crb

            obj_selnet[ss] = objective(F * S_hard.unsqueeze(0), W).item()
            obj_sub[ss] = objective(F * fixed_mask, W).item()
            obj_full[ss] = objective(F, W).item()

        print(f'  J(SelectionNet) = {obj_selnet[ss]:.4f}, '
              f'J(sub-connected) = {obj_sub[ss]:.4f}, '
              f'J(full-connected) = {obj_full[ss]:.4f}')

    # ---- Plot objective vs SNR
    plt.figure(figsize=(8, 5))
    plt.plot(snr_dB_list, obj_selnet, '-o', color='red', linewidth=3, markersize=8,
             label='SelectionNet (learned mask)')
    plt.plot(snr_dB_list, obj_sub, '--s', color='blue', linewidth=3, markersize=8,
             label='Fixed sub-connected')
    plt.plot(snr_dB_list, obj_full, '-d', color='black', linewidth=3, markersize=8,
             label='Full-connected')
    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel(r'$J = \omega R + \log(\mathrm{CRLB}^{-1})$', fontsize=14)
    plt.title('Objective vs SNR: SelectionNet vs fixed sub-connected vs full-connected')
    plt.grid()
    safe_legend(loc='best', fontsize=12, labelspacing=0.15)
    plt.tight_layout()
    plt.savefig(directory_result + 'objective_vs_SNR_SelectionNet_' + str(Nt) + '_' + str(OMEGA) + '.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'objective_vs_SNR_SelectionNet_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
    plt.show()
    print('Saved objective-vs-SNR figure to ' + directory_result)


# ============================================================== main =================================
if __name__ == "__main__":
    if run_SelectionNet == 1:
        run_selectionnet()
