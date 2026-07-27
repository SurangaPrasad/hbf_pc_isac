"""
MA-FAHP: Matching-Assisted Fully-Adaptive Hybrid Precoding
Xue et al., "Energy-Efficient Hybrid Precoding for Massive MIMO mmWave
Systems With a Fully-Adaptive-Connected Structure," IEEE TCOM 2020.

This module adapts Algorithm 2 (the matching-based search over RF-chain /
antenna connection-state matrices D) of the paper to this project's trained
deep-unfolded PGA model. Instead of the paper's own AHP/Dinkelbach solver,
the CHP subproblem (i.e. computing the analog/digital precoders for a given
D) is solved by a forward pass of the already-trained `PGA_Unfold_JX_partial`
(UPGA-PC) model, and the resulting objective (Eq. 10/11, matching the loss
the model was trained with -- see `get_sum_loss` in PGA_models.py) is used as
the matching utility U(D).

Implements:
  - UPGA-PC-based objective / utility function             (Eq. 10, 11, 26)
  - MA-FAHP algorithm (Algorithm 2)                          (Definitions 1-4)
"""

from PGA_models import *

# --------------------------------------------------------------------------
# 0. Problem parameters container
# --------------------------------------------------------------------------
class Params:
    """
    Bundles everything `calculate_utility` needs to score a connection-state
    matrix D using the trained UPGA-PC model as the CHP-subproblem solver.
    """
    def __init__(self, H, Pt, model,
                 n_iter_outer_search=10, n_iter_outer_eval=n_iter_outer,
                 H_eval=None, p_i=None, q_j=None):
        self.H = H                      # (K, B, M, Nt) channel batch used while *searching* for D
        self.H_eval = H_eval if H_eval is not None else H  # batch used for the *final* reported objective
        self.Pt = Pt                    # linear transmit power (SNR)
        self.model = model              # trained PGA_Unfold_JX_partial (frozen weights)
        self.M_T, self.N_RF = Nt, Nrf   # kept for compatibility with the matching helpers below
        self.n_iter_outer_search = n_iter_outer_search  # cheap budget used while searching for D
        self.n_iter_outer_eval = n_iter_outer_eval       # full budget used for the final reported objective
        self.p_i = p_i if p_i is not None else np.full(Nt, Nrf // 2)   # (12e)
        self.q_j = q_j if q_j is not None else np.full(Nrf, Nt // 4)   # (12f)


# --------------------------------------------------------------------------
# 1. CHP subproblem solver: forward pass of the trained UPGA-PC model
# --------------------------------------------------------------------------
def load_pga_partial_model(step_size, model_path, mask=None):
    """
    Build a `PGA_Unfold_JX_partial` model (Nt/Nrf from system_config.py) and
    load its trained (step-size) weights.

    `strict=False` because the connection `mask` buffer is overwritten
    per-candidate D by `_set_mask` before every forward pass, so whatever
    'mask' entry (if any) is stored in the checkpoint is irrelevant here.
    """
    model = PGA_Unfold_JX_partial(step_size, Nt=Nt, Nrf=Nrf, mask=mask)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _set_mask(model, D):
    """Overwrite `model`'s connection-state buffer with candidate D (Nt, Nrf numpy 0/1 array)."""
    mask = torch.as_tensor(D, dtype=REAL_DTYPE, device=device)
    model.mask = mask
    return mask


def evaluate_configuration(D, params, H, n_iter_outer):
    """
    Run the UPGA-PC model with connection matrix D on channel batch `H` for
    `n_iter_outer` outer iterations and return
    (utility, sum_rate, mean_crb, mean_power).

    utility = OMEGA * sum_rate + mean_crb - mean_power, i.e. the same
    objective the model was trained with (see `get_sum_loss` in
    PGA_models.py, negated), Eq. (10)/(11), now with the D-dependent
    circuit power included in `mean_power` via `compute_total_power`.
    """
    mask = _set_mask(params.model, D)

    with torch.no_grad():
        _, _, F, W, _, _ = params.model.execute_PGA(
            H, xi_0, A_dot, R_N_inv, params.Pt,
            n_iter_outer, n_iter_inner_J5, track_metrics=False)

        sum_rate = get_sum_rate(H, F, W, params.Pt)
        mean_crb = torch.mean(get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, params.Pt))
        mean_power = torch.mean(compute_total_power(F, W, mask, P_RF, P_PS, P_SW, P_o, Nt, Nrf))

        utility = (OMEGA * sum_rate + mean_crb - mean_power).item()

    return utility, sum_rate.item(), mean_crb.item(), mean_power.item()


def calculate_utility(D, params, cache=None):
    """
    U(D), Eq. (26): objective achieved by the UPGA-PC model under connection
    state D, evaluated on the (small/fast) search batch `params.H` with
    `params.n_iter_outer_search` outer iterations.

    `cache`: optional dict {D.tobytes(): utility} to avoid recomputation.
    """
    key = D.tobytes() if cache is not None else None
    if cache is not None and key in cache:
        return cache[key]

    utility, *_ = evaluate_configuration(D, params, params.H, params.n_iter_outer_search)

    if cache is not None:
        cache[key] = utility
    return utility


# --------------------------------------------------------------------------
# 5. Matching primitives  (Definitions 1-3)
# --------------------------------------------------------------------------
def generate_random_binary_matrix(M_T, N_RF, p_i, q_j, rng=None):
    """Random feasible D satisfying |Ψ(i)|<=p_i and |Ψ(j)|<=q_j (best-effort)."""
    rng = rng or np.random.default_rng()
    D = np.zeros((M_T, N_RF), dtype=int)
    rf_load = np.zeros(N_RF, dtype=int)

    for i in range(M_T):
        n_conn = rng.integers(1, p_i[i] + 1)
        candidates = [j for j in range(N_RF) if rf_load[j] < q_j[j]]
        rng.shuffle(candidates)
        chosen = candidates[:n_conn]
        for j in chosen:
            D[i, j] = 1
            rf_load[j] += 1
    return D


def active_connections(D, i):
    """Ψ(i): RF chains currently connected to antenna i."""
    return list(np.nonzero(D[i, :])[0])


def available_rf_chains(D, i, params):
    """
    RF chains j NOT connected to antenna i, for which both antenna i's
    degree constraint (p_i) and RF chain j's degree constraint (q_j) allow
    a new connection.
    """
    if np.sum(D[i, :]) >= params.p_i[i]:
        return []
    rf_load = np.sum(D, axis=0)
    return [j for j in range(params.N_RF)
            if D[i, j] == 0 and rf_load[j] < params.q_j[j]]


def perform_swap(D, i, j, i_prime, j_prime):
    """Definition 1: antenna i <-> antenna i' swap RF chains j <-> j'."""
    D_new = D.copy()
    D_new[i, j] = 0
    D_new[i, j_prime] = 1
    D_new[i_prime, j_prime] = 0
    D_new[i_prime, j] = 1
    return D_new


def perform_join(D, i, j):
    """Definition 2: connect antenna i to RF chain j."""
    D_new = D.copy()
    D_new[i, j] = 1
    return D_new


def perform_leave(D, i, j):
    """Definition 3: disconnect antenna i from RF chain j."""
    D_new = D.copy()
    D_new[i, j] = 0
    return D_new


# --------------------------------------------------------------------------
# 6. MA-FAHP main algorithm (Algorithm 2)
# --------------------------------------------------------------------------
def calculate_final_precoders(D, params):
    """Run the UPGA-PC model with the final D on the (full) evaluation batch/budget."""
    _set_mask(params.model, D)
    with torch.no_grad():
        _, _, F, W, _, _ = params.model.execute_PGA(
            params.H_eval, xi_0, A_dot, R_N_inv, params.Pt,
            params.n_iter_outer_eval, n_iter_inner_J5, track_metrics=False)
    return F, W


def ma_fahp(params, D_init=None, max_while_loops=100, verbose=True, rng=None):
    """
    Algorithm 2. Returns the converged connection-state matrix D.
    Call `calculate_final_precoders(D, params)` afterwards to obtain the
    corresponding (F, W) on the full evaluation batch/iteration budget.
    """
    rng = rng or np.random.default_rng()
    D = D_init if D_init is not None else generate_random_binary_matrix(
        params.M_T, params.N_RF, params.p_i, params.q_j, rng)

    cache = {}                        # avoids recomputing utility for repeated D
    stability_indicator = 0
    loop_count = 0

    while stability_indicator == 0 and loop_count < max_while_loops:
        loop_count += 1
        D_previous = D.copy()

        for i in range(params.M_T):
            U_current = calculate_utility(D, params, cache)

            # ---------------- SWAP PHASE (Eq. 30, 33) ----------------
            best_opu, best_D = 0.0, None
            for i_prime in range(params.M_T):
                if i_prime == i:
                    continue
                js = active_connections(D, i)
                j_primes = available_rf_chains(D, i_prime, params)
                for j in js:
                    # j' must not already be connected to i, per Definition 1
                    for j_prime in j_primes:
                        if D[i, j_prime] == 1 or D[i_prime, j] == 1:
                            continue
                        D_temp = perform_swap(D, i, j, i_prime, j_prime)
                        opu = max(0.0, calculate_utility(D_temp, params, cache) - U_current)
                        if opu > best_opu:
                            best_opu, best_D = opu, D_temp
            if best_opu > 0:
                D = best_D
                U_current = calculate_utility(D, params, cache)

            # ---------------- JOINING-IN PHASE (Eq. 31, 34) ----------------
            best_opu, best_D = 0.0, None
            for j in available_rf_chains(D, i, params):
                D_temp = perform_join(D, i, j)
                opu = max(0.0, calculate_utility(D_temp, params, cache) - U_current)
                if opu > best_opu:
                    best_opu, best_D = opu, D_temp
            if best_opu > 0:
                D = best_D
                U_current = calculate_utility(D, params, cache)

            # ---------------- LEAVING PHASE (Eq. 32, 35) ----------------
            best_opu, best_D = 0.0, None
            for j in active_connections(D, i):
                D_temp = perform_leave(D, i, j)
                opu = max(0.0, calculate_utility(D_temp, params, cache) - U_current)
                if opu > best_opu:
                    best_opu, best_D = opu, D_temp
            if best_opu > 0:
                D = best_D

        # ---------------- Convergence check ----------------
        if np.array_equal(D, D_previous):
            stability_indicator = 1

        if verbose:
            U = calculate_utility(D, params, cache)
            print(f"[MA-FAHP] while-loop {loop_count}: objective = {U:.4f}, "
                  f"active links = {int(D.sum())}")

    return D