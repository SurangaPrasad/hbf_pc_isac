# Training SelectionNet

Learnable antenna-to-RF-chain assignment for the sub-connected mmWave massive MIMO ISAC hybrid beamforming system.

## What it does

`SelectionNet` (`SelectionNet.py`) is an MLP that maps the channel `H` and a sensing
direction `psi0` to a connection matrix `S` of shape `(batch, Nt, Nrf)`. Each row of
`S` is a soft/hard distribution over the RF chains: antenna `n` is assigned to the RF
chain with the largest entry. This replaces the fixed hand-built sub-connected mask
(e.g. the template in `PGA_models.py:406-428`) with a **learned, per-sample mask**.

Because there is no ground-truth assignment, the network is trained with the same
**unsupervised physics loss** used for the UPGA step sizes:

```
loss = get_sum_loss(F, W, H, ...) = -(OMEGA * sum_rate + mean_CRB)
```

## The training flow (main_selection.py)

The training loop lives in `main_selection.py`, function `run_selectionnet()`.
Step by step:

1. **Load data.** `H_train` has shape `(K, B, M, Nt)` (K = frequency bands, B = batch,
   M = users, Nt = antennas), confirmed from `matlab/comm_data.m`.

2. **Build a frozen beamformer.** `load_pretrained_upga()` re-instantiates the exact
   `PGA_Unfold_JX` class with the same `step_size` shape used in its own training
   (`(J, n_iter_outer, K+1)`), loads `model/64TX_4UE_4RF/UPGA_J5.pth`, sets
   `requires_grad_(False)` and `eval()`. This gives the fixed F, W that the loss needs.

3. **Per batch:**
   - Take channels `H_batch` of shape `(K, B, M, Nt)`.
   - Reformat for SelectionNet: `H_sel = H_batch[0].transpose(1, 2)` → `(B, Nt, M)`.
   - `psi0` is the fixed sensing direction `desired_angle_rad` replicated over the batch.
   - **SelectionNet forward:** `S, logits = selnet(H_sel, psi0, tau, hard=False)`.
     During training `hard=False` (soft Gumbel-softmax), so gradients can flow.
     `tau` decays exponentially over epochs (`anneal_tau`, 2.0 → 0.1) so the mask
     sharpens toward one-hot as training progresses.
   - **Physics forward (no grad):** the frozen UPGA runs `execute_PGA(...)` to produce
     `F` of shape `(K, B, Nt, Nrf)` and `W` of shape `(K, B, Nrf, M)`.
   - **Apply the learned mask:** `F_eff = F * S.unsqueeze(0)` broadcasts `S` over the
     K dimension, zeroing every antenna→RF-chain path the network did not select.
   - **Loss:** `loss = get_sum_loss(F_eff, W, H_batch, ...)` evaluates the physical
     sum-rate + CRB objective with the masked precoder.
   - **Backprop:** `loss.backward()` — the gradient path is
     `loss → get_sum_loss → F_eff → S → gumbel_softmax → logits → selnet`.
     Only `selnet.parameters()` are updated; F, W are detached by the `no_grad` block.
   - Optionally add a load-balancing regularizer:
     `loss += SELNET_LOAD_BALANCE_WEIGHT * selnet.column_load(S).std(dim=-1).mean()`
     (off by default, `SELNET_LOAD_BALANCE_WEIGHT = 0.0`).

4. **Save & plot.** After all epochs, the state dict is saved as
   `SelectionNet_J5.pth` in `model/64TX_4UE_4RF/`, and a loss-vs-epoch curve is written
   to `SelectionNet_loss.png`.

5. **Sanity check.** A hard-mode (`hard=True`) forward on a small batch prints the
   number of antennas assigned to each RF chain and verifies the total equals `Nt`.

6. **Objective-vs-SNR evaluation.** `plot_selectionnet_objective_vs_snr()` sweeps the
   SNR points in `snr_dB_list` and plots the physics objective
   `J = OMEGA * R + mean(log CRLB)` for three schemes that share the *same* frozen
   UPGA J5 beamformer (F, W) and differ only in the antenna→RF-chain mask applied to F:
   - **SelectionNet** — per-sample learned mask `S_hard` from the trained network
     (hard one-hot mode).
   - **Fixed sub-connected** — the uniform block mask `generage_partial_connection_mask`
     (`Nt/Nrf` antennas per RF chain).
   - **Full-connected** — no mask at all (F is used directly).
   The figure is written to
   `sim_results/64TX_4UE_4RF/objective_vs_SNR_SelectionNet_64_0.25.png/.eps`.

### Gradient flow (why it works)

```
loss ──> get_sum_loss ──> F_eff = F * S ──> S (gumbel-softmax, soft)
   │                                              │
   └── frozen UPGA (no_grad) ───> F, W            └──> selnet parameters (trained)
```

- The Gumbel-softmax makes the discrete assignment differentiable via the
  reparameterization trick (Gumbel noise added to logits, then row-wise softmax).
- The straight-through estimator in `hard` mode returns one-hot values on the forward
  pass but keeps soft-probability gradients on the backward pass.
- Since the UPGA is frozen, only the assignment network learns — the beamformer itself
  is fixed and the network only reshuffles which antennas connect to which RF chains.

## How to run it

Requirements: Python 3.10+, PyTorch 2.x, and the existing repo data
(`dataset/64TX_4UE_4RF/train_data_matlab.mat`) plus a pre-trained UPGA checkpoint
(`model/64TX_4UE_4RF/UPGA_J5.pth`).

1. **Enable the run flag.** In `system_config.py`, make sure:
   ```python
   run_SelectionNet = 1
   ```
   (Already set to 1. Set to 0 to skip SelectionNet training.)

2. **Optionally tune hyper-parameters** at the top of `main_selection.py`:
   ```python
   SELNET_LR = 1e-3        # Adam learning rate for the assignment network
   SELNET_TAU_START = 2.0  # Gumbel temperature at epoch 0 (exploration)
   SELNET_TAU_END = 0.1    # Gumbel temperature at the last epoch (sharpen)
   SELNET_LOAD_BALANCE_WEIGHT = 0.0  # >0 enables the load-balancing regularizer
   ```
   Epochs, batch size, learning rate etc. come from `system_config.py`
   (`n_epoch`, `batch_size`, `learning_rate`).

3. **Run the script:**
   ```bash
   python main_selection.py
   ```
   Recommended (safe) run:
   ```bash
   python main_selection.py 2>&1 | tee selection_train.log
   ```
   Note: training runs the full physics forward pass per batch (120 outer × 5 inner
   UPGA iterations), so it is expensive. On a machine without a GPU, reduce
   `n_epoch`, `n_iter_outer`, or `n_iter_inner_J5` in `system_config.py`, and ensure
   `device` resolves to CPU (printed at import from `system_config.py:11`).

4. **Outputs.**
   - `model/64TX_4UE_4RF/SelectionNet_J5.pth` — trained weights (state dict)
   - `model/64TX_4UE_4RF/SelectionNet_loss.png` — loss curve
   - `sim_results/64TX_4UE_4RF/objective_vs_SNR_SelectionNet_64_0.25.png/.eps` —
     post-training objective-vs-SNR comparison (SelectionNet vs fixed
     sub-connected vs full-connected)
   - `selection_train.log` — console output (if you used the `tee` command)

## Files involved

| File                | Role                                                                  |
|---------------------|-----------------------------------------------------------------------|
| `SelectionNet.py`   | The MLP + Gumbel-softmax discretization (module only, no loss)         |
| `main_selection.py` | The unsupervised training loop                                         |
| `system_config.py`  | `run_SelectionNet` flag, system & training hyper-parameters            |
| `PGA_models.py`     | `PGA_Unfold_JX` (frozen beamformer) and `get_sum_loss` (physics loss)   |
| `utility.py`        | `get_data_tensor` (data loading), `get_sum_rate` / `get_crb_fe`         |
