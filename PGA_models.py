import math
import torch
import torch.nn as nn
from utility import *
from torch.utils.checkpoint import checkpoint

def clamp_complex_magnitude(delta, max_magnitude):
    """Scale complex updates so their magnitude never exceeds `max_magnitude`."""
    magnitude = torch.abs(delta)
    scale = torch.clamp(max_magnitude / (magnitude + 1e-12), max=1.0)
    return delta * scale


def sanitize_complex_tensor(tensor):
    """Replace NaN/Inf entries in a complex tensor with safe finite values."""
    real = torch.where(torch.isfinite(tensor.real), tensor.real, torch.zeros_like(tensor.real))
    imag = torch.where(torch.isfinite(tensor.imag), tensor.imag, torch.zeros_like(tensor.imag))
    return torch.complex(real, imag)


def project_unit_modulus(F, eps=1e-12, active_mask=None):
    """Project complex entries to unit modulus without introducing NaNs."""
    magnitude = torch.abs(F)
    safe_magnitude = torch.where(magnitude > eps, magnitude, torch.ones_like(magnitude))
    projected = F / safe_magnitude
    projected = torch.where(magnitude > eps, projected, torch.zeros_like(F))

    if active_mask is not None:
        mask = active_mask
        while mask.dim() < F.dim():
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=F.dtype, device=F.device)
        mask_bool = mask.abs() > 0

        # Rebuild the phase for entries that were driven below eps inside the active region.
        phase = torch.polar(torch.ones_like(F.real), torch.angle(F))
        projected = torch.where(mask_bool & (magnitude <= eps), phase, projected)
        projected = torch.where(mask_bool, projected, torch.zeros_like(projected))

    return projected

# /////////////////////////////////////////////////////////////////////////////////////////
#                             PGA MODEL CLASSES
# /////////////////////////////////////////////////////////////////////////////////////////

#  ================================ PGA conventional with different inner iterations ===========================================
class PGA_Conv_comp_grad(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner, weight_grad_F_rad, init_method):
        rate_init, tau_init, F, W = initialize_schemes(H, R, Pt, init_method)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)   # save beampattern errors over iterations
        # update F and W over iterations
        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                # update F
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                # self.step_size[ii][0]
                delta_F_com = self.step_size[ii][0] * grad_F_com
                delta_F_rad = self.step_size[ii][0] * grad_F_rad
                F = F + delta_F_com - delta_F_rad * OMEGA
                # normalize by power to ensure non-NaN gradients if F becomes too large
                if sum(torch.abs(F[0, :, 0, 0])) > 1e1:
                    F = normalize_power(F, W, H, Pt)
                # Projection
                F = project_unit_modulus(F)

                # update W
                W_new = W.clone().detach()
                # compute gradients
                grad_W_k_com = get_grad_W_com(H, F, W)
                grad_W_k_rad = get_grad_W_rad(F, W, R)
                for k in range(K):
                    delta_W_com = self.step_size[ii][k + 1] * grad_W_k_com[k]
                    delta_W_rad = self.step_size[ii][k + 1] * grad_W_k_rad[k]
                    W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD

            # projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)
            # print(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
        return torch.transpose(rates,  0, 1), torch.transpose(taus,  0, 1), F, W

#  ================================ UPGA with J = 1 and conventional ===========================================
class PGA_Conv(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, track_metrics=True):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, len(H[0]), device=H.device)
        # update F and W over iterations
        for ii in range(n_iter_outer):
            # update F
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_rad = get_grad_F_rad(F, W, R)
            delta_F_com = self.step_size[ii][0] * grad_F_com
            delta_F_rad = self.step_size[ii][0] * grad_F_rad
            F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD

            # Projection
            F = project_unit_modulus(F)

            # update W  (K == 1 always, unroll the k-loop)
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            W_new = W.clone().detach()
            W_new[0] = W[0].detach() + (self.step_size[ii][1] * grad_W_k_com[0]) * WEIGHT_W_COM \
                                     - (self.step_size[ii][1] * grad_W_k_rad[0]) * WEIGHT_W_RAD

            # projection
            F, W = normalize(F, W_new, H, Pt)

            # per-iteration metrics (skip during training for speed)
            if track_metrics:
                rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
                tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W

# ============================================== Proposed PGA model=============================

class PGA_Unfold_JX(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner, track_metrics=True):
        rate_init, F, W = initialize(H, Pt, initial_normalization)
        B = len(H[0])
        # Shape: (n_outer, J+1, B)
        #   [:, 0, :] = metrics after W-update (start of each outer iter)
        #   [:, 1..J, :] = metrics after each inner F-update
        rate_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters  = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt):
            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb
                F = F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB
                F = normalize_power(F, W, H, Pt)
            return F

        for ii in range(n_iter_outer):
            if track_metrics:
                # Run inner loop without checkpoint so we can record per-inner metrics
                for jj in range(n_iter_inner):
                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                    delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                    delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb
                    F = F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB
                    F = normalize_power(F, W, H, Pt)  # scale F only, consistent with training path
                    rate_over_iters[ii, jj] = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj]  = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()
            else:
                F = checkpoint(inner_f_update, F, W, H, xi_0, A_dot, R_N_inv, n_iter_inner, Pt, use_reentrant=False)
            F = project_unit_modulus(F)

            # update W  (K == 1 always, unroll the k-loop)
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            W_new = W.clone().detach()
            W_new[0] = W[0].detach() + (self.step_size[0][ii][1] * grad_W_k_com[0]) * WEIGHT_W_COM \
                                     + (self.step_size[0][ii][1] * grad_W_k_crb[0]) * WEIGHT_W_CRB

            # Projection
            F , W = normalize(F, W_new, H, Pt)

            # Record metrics after W-update (slot 0 of this outer iter)
            if track_metrics:
                rate_over_iters[ii, -1] = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, -1]  = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                power_over_iters[ii, -1] = get_power(F, W).detach()

        # Flatten to (n_outer*(J+1), B) then transpose to (B, n_outer*(J+1))
        rates   = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        crb_fes = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        power_fes = power_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        return rates.transpose(0, 1), crb_fes.transpose(0, 1), power_fes.transpose(0, 1), F, W

# ============================================== Unfolded PGA with decaying inner iterations ==============================
class PGA_Unfold_J10_decay(nn.Module):

    def __init__(self, step_size, alpha=1e-4):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)
        self.inner_iter_history = []

        # Adaptive scheduling hyperparameter
        self.alpha = alpha

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner, track_metrics=True):
        rate_init, F, W = initialize(H, Pt, initial_normalization)
        B = len(H[0])
        # Shape: (n_outer, J+1, B)
        #   [:, 0, :] = metrics after W-update (start of each outer iter)
        #   [:, 1..J, :] = metrics after each inner F-update
        rate_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters  = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        def _n_inner(prev_obj=None, curr_obj=None):

            max_inner = self.step_size.shape[0]

            # First outer iteration
            if prev_obj is None or curr_obj is None:
                return max_inner

            eps = 1e-8

            # Relative objective improvement
            delta = torch.mean(
                torch.abs(curr_obj - prev_obj) /
                (torch.abs(prev_obj) + eps)
            )
            # print(f'current_obj: {curr_obj.mean().item()}, prev_obj: {prev_obj.mean().item()}, delta: {delta.item()}')
            # print(f"Delta: {delta.item()}")

            # Adaptive ratio
            ratio = delta / (delta + self.alpha)

            # Adaptive inner iteration count
            n_inner = int(torch.ceil(max_inner * ratio).item())

            # Keep at least 2 iterations for stability
            return max(2, min(max_inner, n_inner))

        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt):

            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb
                F = F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB
                F = normalize_power(F, W, H, Pt)
            return F
        inner_iter_history = []
        for ii in range(n_iter_outer):
            
            # ----------------------------------------------------
            # Adaptive inner iteration count
            # ----------------------------------------------------
            if ii < 2:
                prev_obj = None
                curr_obj = None
            else:
                prev_obj = (OMEGA * rate_over_iters[ii-2, -1] + crb_over_iters[ii-2, -1]).detach()
                curr_obj = (OMEGA * rate_over_iters[ii-1, -1] + crb_over_iters[ii-1, -1]).detach()
 
            n_inner = _n_inner(prev_obj, curr_obj)
            

            if track_metrics:
                inner_iter_history.append(n_inner)

                # Run inner loop without checkpoint so we can record per-inner metrics
                for jj in range(n_inner):
                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                    delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                    delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb
                    F = F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB
                    F = normalize_power(F, W, H, Pt)  # scale F only, consistent with training path
                    rate_over_iters[ii, jj] = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj]  = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()
            else:
                F = checkpoint(inner_f_update, F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt, use_reentrant=False)
            F = project_unit_modulus(F)

            # update W  (K == 1 always, unroll the k-loop)
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            W_new = W.clone().detach()
            W_new[0] = W[0].detach() + (self.step_size[0][ii][1] * grad_W_k_com[0]) * WEIGHT_W_COM \
                                     + (self.step_size[0][ii][1] * grad_W_k_crb[0]) * WEIGHT_W_CRB

            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # Record metrics after W-update (slot 0 of this outer iter)
            if track_metrics:
                rate_over_iters[ii, -1] = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, -1]  = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                power_over_iters[ii, -1] = get_power(F, W).detach()

        if track_metrics:
            rate_slots = []
            crb_slots = []
            power_slots = []
            for ii, n_inner_ii in enumerate(inner_iter_history):
                if n_inner_ii > 0:
                    rate_slots.append(rate_over_iters[ii, :n_inner_ii])
                    crb_slots.append(crb_over_iters[ii, :n_inner_ii])
                    power_slots.append(power_over_iters[ii, :n_inner_ii])
                rate_slots.append(rate_over_iters[ii, -1:].clone())
                crb_slots.append(crb_over_iters[ii, -1:].clone())
                power_slots.append(power_over_iters[ii, -1:].clone())

            rates = torch.cat(rate_slots, dim=0).detach()
            crb_fes = torch.cat(crb_slots, dim=0).detach()
            power_fes = torch.cat(power_slots, dim=0).detach()
        else:
            # No per-inner metrics are tracked on this path, so retain the fixed rectangular layout.
            rates = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
            crb_fes = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
            power_fes = power_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()

        self.inner_iter_history = list(inner_iter_history)
        print("Adaptive inner iterations:", inner_iter_history)
        # print("Average inner iterations:", sum(inner_iter_history) / len(inner_iter_history))

        return rates.transpose(0, 1), crb_fes.transpose(0, 1), power_fes.transpose(0, 1), F, W

# PGA_Unfold_J20_decay is identical to PGA_Unfold_J10_decay; the max inner-iteration
# count is read dynamically from self.step_size.shape[0], so passing a step_size
# tensor with n_iter_inner_J20 rows gives J20 behaviour automatically.
PGA_Unfold_J20_decay = PGA_Unfold_J10_decay

# ============================================ Proposed PGA model with gradient reuse ====================================
class PGA_Unfold_J_GradReuse(nn.Module):
    """Unfolded PGA with lazy gradient reuse to reduce per-inner-iteration cost.

    Uses J = n_iter_inner (default 10) fixed inner iterations per outer iteration.

    F inner-iteration strategy for each outer iteration ``ii``:
      jj = 0  : Always compute fresh gradients (get_grad_F_com + get_grad_F_crb).
      jj >= 1 : Propose F_trial by reusing the last stored gradient with the current
                step size.  Then evaluate the combined objective:
                    obj = sum_rate * WEIGHT_F_COM + mean(crb_fe) * WEIGHT_F_CRB
                - If obj(F_trial) > obj(F_current) → accept F_trial (reuse).
                - Otherwise                        → recompute fresh gradients
                  (fallback), log the recomputation, and step normally.

    The stored F gradient is refreshed only at jj=0 or on a fallback recomputation,
    so a sequence of accepted reuses all share the same fixed gradient direction.

    W-update strategy across outer iterations:
      ii = 0  : Always compute fresh W gradients (get_grad_W_com + get_grad_W_crb).
      ii >= 1 : Propose W_trial by reusing the stored W gradients from the previous
                outer iteration.  Compare against the baseline objective at
                (F_projected, W_current) just before the W step:
                    obj = sum_rate * WEIGHT_W_COM + mean(crb_fe) * WEIGHT_W_CRB
                - If obj(W_trial) > obj(W_current) → accept W_trial (reuse).
                - Otherwise                        → recompute fresh W gradients
                  (fallback), log the recomputation, and step normally.

    The stored W gradient is refreshed only at ii=0 or on a fallback recomputation,
    so consecutive outer iterations may reuse the same W gradient direction.

    Attributes
    ----------
    grad_recalc_count   : total F fallback recomputations (excludes mandatory jj=0)
                          from the most recent ``execute_PGA`` call.
    W_grad_recalc_count : total W fallback recomputations (excludes mandatory ii=0)
                          from the most recent ``execute_PGA`` call.

    step_size shape: [n_iter_inner, n_iter_outer, K+1]  (identical to PGA_Unfold_J10).
    """

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # [n_iter_inner, n_iter_outer, K+1]
        self.grad_recalc_count = 0    # F fallback recomputations; updated by execute_PGA
        self.W_grad_recalc_count = 0  # W fallback recomputations; updated by execute_PGA

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner, track_metrics=True):
        rate_init, F, W = initialize(H, Pt, initial_normalization)
        B = len(H[0])

        # Metric arrays: shape (n_outer, J+1, B).
        #   [ii, 0..J-1, :] – after each inner F-update.
        #   [ii,    J  , :] – after W-update (end of outer iter ii).
        rate_over_iters  = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters   = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        grad_recalc = 0    # F fallback recomputations (excludes mandatory jj=0 gradients)
        W_grad_recalc = 0  # W fallback recomputations (excludes mandatory ii=0 gradients)

        # W gradient state persists across outer iterations (unlike F which resets each outer iter).
        prev_grad_W_k_com = None
        prev_grad_W_k_crb = None

        for ii in range(n_iter_outer):
            prev_grad_F_com = None  # last stored gradient, refreshed at jj=0 or on fallback
            prev_grad_F_crb = None
            prev_obj = None         # Python float: combined objective at current F

            for jj in range(n_iter_inner):
                if jj == 0:
                    # ---- Always compute a fresh gradient at the first inner step ----
                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                    if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                        print('Error NaN gradients!!!!!!!!!!!!!!!')
                else:
                    # # ---- jj >= 1: attempt gradient reuse ----
                    # # Propose a step using the last stored gradient.
                    F_trial = (F
                               + self.step_size[jj][ii][0] * prev_grad_F_com * WEIGHT_F_COM
                               + self.step_size[jj][ii][0] * prev_grad_F_crb * WEIGHT_F_CRB)
                    F_trial = normalize_power(F_trial, W, H, Pt)

                    # Evaluate combined objective comparison (no grad tracking required).
                    with torch.no_grad():
                        r_trial = get_sum_rate(H, F_trial, W, Pt)
                        c_trial = get_crb_fe(H, F_trial, W, xi_0, A_dot, R_N_inv, Pt)
                        obj_trial = (r_trial * WEIGHT_F_COM + c_trial.mean() * WEIGHT_F_CRB).item()

                    if obj_trial > prev_obj:
                        # ---- Reuse accepted ----
                        F = F_trial
                        prev_obj = obj_trial
                        # prev_grad_F_com/crb left unchanged so next jj reuses the same gradient.
                        if track_metrics:
                            rate_over_iters[ii, jj]  = r_trial.detach()
                            crb_over_iters[ii, jj]   = c_trial.detach()
                            power_over_iters[ii, jj] = get_power(F, W).detach()
                        continue

                    else:
                        # ---- Reuse rejected: recompute gradient from current F ----
                        grad_F_com = get_grad_F_com(H, F, W)
                        grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                        if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                            print('Error NaN gradients!!!!!!!!!!!!!!!')
                        grad_recalc += 1

                # ---- Apply gradient step (jj=0 or reuse-rejected) ----
                F = (F
                     + self.step_size[jj][ii][0] * grad_F_com * WEIGHT_F_COM
                     + self.step_size[jj][ii][0] * grad_F_crb * WEIGHT_F_CRB)
                F = normalize_power(F, W, H, Pt)

                # Store gradient and current objective baseline for next inner iter.
                prev_grad_F_com = grad_F_com.detach()
                prev_grad_F_crb = grad_F_crb.detach()
                with torch.no_grad():
                    r_cur = get_sum_rate(H, F, W, Pt)
                    c_cur = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt)
                    prev_obj = (r_cur * WEIGHT_F_COM + c_cur.mean() * WEIGHT_F_CRB).item()

                if track_metrics:
                    rate_over_iters[ii, jj]  = r_cur.detach()
                    crb_over_iters[ii, jj]   = c_cur.detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()

            F = project_unit_modulus(F)

            # ---- W update with gradient reuse across outer iterations ----
            # Baseline objective at (F_projected, W_current) for the reuse comparison.
            with torch.no_grad():
                r_preW = get_sum_rate(H, F, W, Pt)
                c_preW = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt)
                obj_preW = (r_preW * WEIGHT_W_COM + c_preW.mean() * WEIGHT_W_CRB).item()

            w_reuse_accepted = False
            if ii > 0 and prev_grad_W_k_com is not None:
                # ---- ii >= 1: attempt gradient reuse for W ----
                W_trial_new = W.clone().detach()
                W_trial_new[0] = (W[0].detach()
                                  + (self.step_size[0][ii][1] * prev_grad_W_k_com) * WEIGHT_W_COM
                                  + (self.step_size[0][ii][1] * prev_grad_W_k_crb) * WEIGHT_W_CRB)
                F_wt, W_trial = normalize(F, W_trial_new, H, Pt)

                with torch.no_grad():
                    r_wt = get_sum_rate(H, F_wt, W_trial, Pt)
                    c_wt = get_crb_fe(H, F_wt, W_trial, xi_0, A_dot, R_N_inv, Pt)
                    obj_wt = (r_wt * WEIGHT_W_COM + c_wt.mean() * WEIGHT_W_CRB).item()

                if obj_wt > obj_preW:
                    # ---- W reuse accepted ----
                    F, W = F_wt, W_trial
                    w_reuse_accepted = True
                    # prev_grad_W_k_com/crb left unchanged: next outer iter reuses same gradient.
                    if track_metrics:
                        rate_over_iters[ii, -1]  = r_wt.detach()
                        crb_over_iters[ii, -1]   = c_wt.detach()
                        power_over_iters[ii, -1] = get_power(F, W).detach()
                else:
                    # ---- W reuse rejected: recompute W gradients from current (F, W) ----
                    grad_W_k_com = get_grad_W_com(H, F, W)
                    grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
                    if grad_W_k_com[0].isnan().any() or grad_W_k_crb[0].isnan().any():
                        print('Error NaN gradients (W)!!!!!!!!!!!!!!!')
                    W_grad_recalc += 1
            else:
                # ---- ii == 0: always compute fresh W gradients ----
                grad_W_k_com = get_grad_W_com(H, F, W)
                grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
                if grad_W_k_com[0].isnan().any() or grad_W_k_crb[0].isnan().any():
                    print('Error NaN gradients (W)!!!!!!!!!!!!!!!')

            if not w_reuse_accepted:
                # ---- Apply W gradient step (ii=0 or reuse-rejected) ----
                W_new = W.clone().detach()
                W_new[0] = (W[0].detach()
                            + (self.step_size[0][ii][1] * grad_W_k_com[0]) * WEIGHT_W_COM
                            + (self.step_size[0][ii][1] * grad_W_k_crb[0]) * WEIGHT_W_CRB)
                F, W = normalize(F, W_new, H, Pt)

                # Store W gradients for reuse in the next outer iteration.
                prev_grad_W_k_com = grad_W_k_com[0].detach()
                prev_grad_W_k_crb = grad_W_k_crb[0].detach()

                if track_metrics:
                    rate_over_iters[ii, -1]  = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, -1]   = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, -1] = get_power(F, W).detach()

        # Log and store gradient recomputation counts.
        self.grad_recalc_count = grad_recalc
        self.W_grad_recalc_count = W_grad_recalc
        max_possible_F = n_iter_outer * (n_iter_inner - 1)
        max_possible_W = n_iter_outer - 1
        print(f'[GradReuse] F fallback recomputations = {grad_recalc} / {max_possible_F} '
              f'({100.0 * grad_recalc / max(max_possible_F, 1):.1f}%)')
        print(f'[GradReuse] W fallback recomputations = {W_grad_recalc} / {max_possible_W} '
              f'({100.0 * W_grad_recalc / max(max_possible_W, 1):.1f}%)')

        # Flatten to (n_outer*(J+1), B) then transpose to (B, n_outer*(J+1)).
        rates     = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        crb_fes   = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        power_fes = power_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        return rates.transpose(0, 1), crb_fes.transpose(0, 1), power_fes.transpose(0, 1), F, W


# ============================================== Proposed PGA model light with preconditioner=============================
class PGA_Unfold_J10_PRCDN(nn.Module):

    def __init__(self, n_iter_inner, n_iter_outer, dim_F, dim_W):
        super().__init__()

        # ===== Diagonal preconditioner for F =====
        # Shape: [n_iter_inner, n_iter_outer, 64]
        self.mu = nn.Parameter( 1e-2 * torch.ones(n_iter_inner, n_iter_outer, dim_F, device=device))

        # ===== Diagonal preconditioner for W =====
        # Shape: [n_iter_outer, 4]
        self.lambda_ = nn.Parameter( 1e-2 * torch.ones(n_iter_outer, dim_W, device=device))



    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner, track_metrics=True):
        rate_init, F, W = initialize(H, Pt, initial_normalization)
        B = len(H[0])
        # Shape: (n_outer, J+1, B)
        rate_over_iters  = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters   = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, mu_ii, n_inner, Pt):
            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                grad_vec_com = grad_F_com.reshape(64, -1)
                grad_vec_crb = grad_F_crb.reshape(64, -1)
                mu_vec = mu_ii[jj]
                delta_vec_com = mu_vec.unsqueeze(1) * grad_vec_com
                delta_vec_crb = mu_vec.unsqueeze(1) * grad_vec_crb
                delta_F_com = delta_vec_com.reshape_as(grad_F_com)
                delta_F_crb = delta_vec_crb.reshape_as(grad_F_crb)
                F = F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB
                F = normalize_power(F, W, H, Pt)
            return F

        for ii in range(n_iter_outer):
            if track_metrics:
                # Run inner loop without checkpoint so we can record per-inner metrics
                mu_ii = self.mu[:, ii]
                for jj in range(n_iter_inner):
                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                    grad_vec_com = grad_F_com.reshape(64, -1)
                    grad_vec_crb = grad_F_crb.reshape(64, -1)
                    mu_vec = mu_ii[jj]
                    delta_F_com = (mu_vec.unsqueeze(1) * grad_vec_com).reshape_as(grad_F_com)
                    delta_F_crb = (mu_vec.unsqueeze(1) * grad_vec_crb).reshape_as(grad_F_crb)
                    F = F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB
                    F = normalize_power(F, W, H, Pt)
                    rate_over_iters[ii, jj]  = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj]   = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()
            else:
                F = checkpoint(inner_f_update, F, W, H, xi_0, A_dot, R_N_inv, self.mu[:, ii], n_iter_inner, Pt, use_reentrant=False)
            F = project_unit_modulus(F)

            # update W  (K == 1 always, unroll the k-loop)
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            lambda_vec = self.lambda_[ii]
            grad_vec_com = grad_W_k_com[0].reshape(4, -1)
            grad_vec_crb = grad_W_k_crb[0].reshape(4, -1)
            delta_vec_com = lambda_vec.unsqueeze(1) * grad_vec_com
            delta_vec_crb = lambda_vec.unsqueeze(1) * grad_vec_crb
            W_new = W.clone().detach()
            W_new[0] = W[0] + delta_vec_com.reshape_as(grad_W_k_com[0]) * WEIGHT_W_COM \
                            + delta_vec_crb.reshape_as(grad_W_k_crb[0]) * WEIGHT_W_CRB

            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # Record metrics after W-update (last slot of this outer iter)
            if track_metrics:
                rate_over_iters[ii, -1]  = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, -1]   = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                power_over_iters[ii, -1] = get_power(F, W).detach()

        # Flatten to (n_outer*(J+1), B) then transpose to (B, n_outer*(J+1))
        rates     = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        crb_fes   = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        power_fes = power_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        return rates.transpose(0, 1), crb_fes.transpose(0, 1), power_fes.transpose(0, 1), F, W

# ============================================== Proposed PGA model light for RMSProp=============================

class PGA_Unfold_J10_RMSProp(nn.Module):

    def __init__(self):
        super().__init__()

        self.beta = 0.9
        self.eps = 1e-8
        self.eta_F = 1e-2   # base learning rate for F
        self.eta_W = 1e-2   # base learning rate for W



    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner, track_metrics=True):
        rate_init, F, W = initialize(H, Pt, initial_normalization)
        B = len(H[0])
        # Shape: (n_outer, J+1, B)
        rate_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters  = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        s_F = torch.zeros_like(F)
        s_W = torch.zeros_like(W)

        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt, s_F):
            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                grad_F = grad_F_com * WEIGHT_F_COM + grad_F_crb * WEIGHT_F_CRB
                s_F = self.beta * s_F + (1 - self.beta) * grad_F
                F = F + self.eta_F * grad_F / (torch.sqrt(s_F) + self.eps)
                F = normalize_power(F, W, H, Pt)
            return F

        for ii in range(n_iter_outer):
            if track_metrics:
                # Run inner loop without checkpoint so we can record per-inner metrics
                for jj in range(n_iter_inner):
                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                    grad_F = grad_F_com * WEIGHT_F_COM + grad_F_crb * WEIGHT_F_CRB
                    s_F = self.beta * s_F + (1 - self.beta) * grad_F
                    F = F + self.eta_F * grad_F / (torch.sqrt(s_F) + self.eps)
                    F = normalize_power(F, W, H, Pt)
                    rate_over_iters[ii, jj + 1] = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj + 1]  = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
            else:
                F = checkpoint(inner_f_update, F, W, H, xi_0, A_dot, R_N_inv, n_iter_inner, Pt, s_F, use_reentrant=False)
            F = project_unit_modulus(F)

            # update W with RMSProp (K == 1 always, unroll the k-loop)
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            if ii == 0:
                s_W = torch.zeros_like(W)
            grad_W_0 = grad_W_k_com[0] * WEIGHT_W_COM + grad_W_k_crb[0] * WEIGHT_W_CRB
            s_W[0] = self.beta * s_W[0] + (1 - self.beta) * grad_W_0
            W_new = W.clone()
            W_new[0] = W[0] + self.eta_W * grad_W_0 / (torch.sqrt(s_W[0]) + self.eps)
            F, W = normalize(F, W_new, H, Pt)

            # Record metrics after W-update (slot 0 of this outer iter)
            if track_metrics:
                rate_over_iters[ii, 0] = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, 0]  = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()

        # Flatten to (n_outer*(J+1), B) then transpose to (B, n_outer*(J+1))
        rates   = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        crb_fes = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        return rates.transpose(0, 1), crb_fes.transpose(0, 1), F, W

# /////////////////////////////////////////////////////////////////////////////////////////
#                             COMM GRADIENTS
# /////////////////////////////////////////////////////////////////////////////////////////


# ==================================== gradient of R_mk w.r.t. F ===========================
def get_grad_F_com(H, F, W):
    """Vectorised gradient of sum-rate w.r.t. F (no Python loop over users)."""
    F_H = F.conj().transpose(-2, -1)          # (K, B, Nrf, Nt)
    W_H = W.conj().transpose(-2, -1)           # (K, B, M, Nrf)
    V   = W @ W_H                               # (K, B, Nrf, Nrf)
    K_d = W.shape[0]

    # Per-user outer products w_m w_m^H -> V_mk = V - outer_m
    # w_cols: (K, B, M, Nrf, 1)
    w_cols = W.permute(0, 1, 3, 2).unsqueeze(-1)
    V_m    = w_cols @ w_cols.conj().transpose(-2, -1)   # (K, B, M, Nrf, Nrf)
    V_mk   = V.unsqueeze(2) - V_m                        # (K, B, M, Nrf, Nrf)

    # Channel outer products H_tilde_m = h_m h_m^H  (K, B, M, Nt, Nt)
    h       = H.unsqueeze(-1)                             # (K, B, M, Nt, 1)
    Htilde  = h @ h.conj().transpose(-2, -1)              # (K, B, M, Nt, Nt)

    # Shared: F @ V @ F_H                                 (K, B, Nt, Nt)
    FVF_H = F @ V @ F_H

    # Quadratic forms via h^H A h  (cheap compared to full Nt×Nt trace)
    qf1 = (h.conj().transpose(-2, -1) @ FVF_H.unsqueeze(2) @ h).squeeze(-1).squeeze(-1)  # (K,B,M)
    denom1 = np.log(2) * (qf1 + sigma2)

    FVmk    = F.unsqueeze(2) @ V_mk                       # (K, B, M, Nt, Nrf)
    FVmkF_H = FVmk @ F_H.unsqueeze(2)                     # (K, B, M, Nt, Nt)
    qf2 = (h.conj().transpose(-2, -1) @ FVmkF_H @ h).squeeze(-1).squeeze(-1)  # (K,B,M)
    denom2 = np.log(2) * (qf2 + sigma2)

    HtF   = Htilde @ F.unsqueeze(2)                        # (K, B, M, Nt, Nrf)
    grad1 = HtF @ V.unsqueeze(2)  / (denom1.unsqueeze(-1).unsqueeze(-1) + 1e-4)  # (K,B,M,Nt,Nrf)
    grad2 = HtF @ V_mk            / (denom2.unsqueeze(-1).unsqueeze(-1) + 1e-4)  # (K,B,M,Nt,Nrf)

    # Sum over M users, average over K frequencies
    grad_F = (grad1 - grad2).sum(dim=2) / K_d             # (K, B, Nt, Nrf)
    return grad_F

def get_grad_W_com(H, F, W):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    V = W @ W_H  # K x train_size x Nrf x Nrf
    grad_W = torch.zeros(len(H), len(H[0]), Nrf, M, dtype=H.dtype, device=H.device)

    for m in range(M):
        W_m = W
        # print(W)
        W_m_H = torch.transpose(W_m, 2, 3).conj()

        h_mk0 = torch.unsqueeze(H[:, :, m, :], dim=2)
        h_mk = torch.transpose(h_mk0, 2, 3)
        h_mk_H = torch.transpose(h_mk, 2, 3).conj()
        Htilde_mk = h_mk @ h_mk_H
        Hbar_mk = F_H @ Htilde_mk @ F

        denom_1 = np.log(2) * (get_trace(W @ W_H @ Hbar_mk) + sigma2)
        grad_W_1 = Hbar_mk @ W / denom_1[:, :, None, None]  # expand dimension

        denom_2 = np.log(2) * (get_trace(W_m @ W_m_H @ Hbar_mk) + sigma2)
        grad_W_2 = Hbar_mk @ W_m / denom_2[:, :, None, None]  # expand dimension
        mask_m = torch.ones(len(H), len(H[0]), Nrf, M, device=H.device)
        mask_m[:, :, :, m] = 0.0
        grad_W_2_masked = grad_W_2 * mask_m  # need element-wise multiplication for masking
        grad_W = grad_W + (grad_W_1 - grad_W_2_masked)

    grad_W = grad_W / K
    return grad_W


def generate_pi_matrix(B_matrix, H_tilde, Nt, Nrf):
    """
    Batched Pi matrix generation for partially connected architecture.
    
    Parameters
    ----------
    B_matrix : torch.Tensor
        Digital covariance matrix, shape (Bsz, Nrf, Nrf)
    H_tilde : torch.Tensor
        Channel outer product, shape (Bsz, Nt, Nt)
    Nt : int
        Total number of antennas
    Nrf : int
        Number of RF chains (subarrays)
    
    Returns
    -------
    Pi : torch.Tensor
        Condensed sensing/channel matrix, shape (Bsz, Nt, Nt)
    """
    Bsz = B_matrix.shape[0]
    antennas_per_rf = Nt // Nrf
    
    Bt = B_matrix.transpose(-1, -2)  # (Bsz, Nrf, Nrf)
    # Reshape H_tilde into (Bsz, Nrf, aprf, Nrf, aprf) blocks, scale by Bt, reshape back
    H_blocks = H_tilde.reshape(Bsz, Nrf, antennas_per_rf, Nrf, antennas_per_rf)
    Pi_blocks = Bt.unsqueeze(2).unsqueeze(4) * H_blocks  # (Bsz, Nrf, aprf, Nrf, aprf)
    Pi = Pi_blocks.reshape(Bsz, Nt, Nt)
    return Pi


def quad_form(Pi, f):
    """
    Computes f^H Pi f for batch.
    
    Parameters
    ----------
    Pi : torch.Tensor
        Matrix, shape (B, Nt, Nt)
    f : torch.Tensor
        Vector, shape (B, Nt, 1)
    
    Returns
    -------
    result : torch.Tensor
        Quadratic form result, shape (B, 1, 1)
    """
    return torch.bmm(f.conj().transpose(-1, -2), torch.bmm(Pi, f))


def extract_active_elements_pc(F, Nt, Nrf):
    """
    Extract active (non-zero) elements from a block-diagonal analog precoder.
    
    Parameters
    ----------
    F : torch.Tensor
        Analog precoder of shape (B, Nt, Nrf), complex
    Nt : int
        Total number of antennas
    Nrf : int
        Number of RF chains (subarrays)
    
    Returns
    -------
    f_active : torch.Tensor
        Active elements of shape (B, Nt, 1)
    """
    B = F.shape[0]
    antennas_per_rf = Nt // Nrf
    
    # Reshape F into blocks: F_blocks[b, m, r, n] = F[b, m*aprf+r, n]
    F_blocks = F.reshape(B, Nrf, antennas_per_rf, Nrf)
    # Extract diagonal blocks: F_diag[b, r, m] = F_blocks[b, m, r, m] = F[b, m*aprf+r, m]
    F_diag = torch.diagonal(F_blocks, dim1=1, dim2=3)  # (B, aprf, Nrf)
    # Rearrange to active vector layout: f_active[b, m*aprf+r, 0] = F[b, m*aprf+r, m]
    f_active = F_diag.permute(0, 2, 1).reshape(B, Nt, 1)
    return f_active


def compute_digital_covariance_pc(W, Pt, Nrf, eps=1e-12):
    """
    Computes the digital precoder covariance matrix V = W W^H
    with PC-specific power normalization.
    
    Parameters
    ----------
    W : torch.Tensor
        Digital precoder, shape (B, Nrf, M)
    Pt : float or torch.Tensor
        Total transmit power (scalar or shape (B,))
    Nrf : int
        Number of RF chains (used for PC normalization)
    eps : float
        Numerical stability constant
    
    Returns
    -------
    V : torch.Tensor
        Digital covariance matrix, shape (B, Nrf, Nrf)
    """
    B = W.shape[0]
    
    # Frobenius norm ||W||_F per batch
    fro_norm = torch.linalg.norm(W, ord='fro', dim=(1, 2), keepdim=True)  # (B, 1, 1)
    
    # Handle scalar or batched Pt
    if not torch.is_tensor(Pt):
        P_vec = torch.tensor(float(Pt), device=W.device).view(1).repeat(B)
    else:
        P_vec = Pt.to(W.device).view(B) if Pt.dim() > 0 else Pt.repeat(B)
    
    # sqrt(Pt / Nrf) for PC normalization
    scale = torch.sqrt(P_vec / Nrf).view(B, 1, 1)
    
    # Normalized W (PC case)
    W_normalized = scale * W / (fro_norm + eps)
    
    # Digital covariance V = W W^H
    V = torch.bmm(W_normalized, W_normalized.conj().transpose(-1, -2))  # (B, Nrf, Nrf)
    
    return V


def get_grad_F_rad_AP(F, W, R, Pt, Nt, Nrf):
    """
    Computes the Euclidean gradient of the sensing objective
    for partially connected architecture.
    
    This is the sensing/radar gradient: ∇_f τ where τ = ||f^H Pi f - target||²
    
    Parameters
    ----------
    F : torch.Tensor
        Analog precoder, shape (K, B, Nt, Nrf) or (B, Nt, Nrf)
    W : torch.Tensor
        Digital precoder, shape (K, B, Nrf, M) or (B, Nrf, M)
    R : torch.Tensor
        Target sensing covariance matrix, shape (K, B, Nt, Nt) or (B, Nt, Nt)
    Pt : float or torch.Tensor
        Total transmit power
    Nt : int
        Total number of antennas
    Nrf : int
        Number of RF chains
    
    Returns
    -------
    grad_F : torch.Tensor
        Gradient w.r.t. F in active element space, same shape as F
    """
    # Handle both 4D (K, B, Nt, Nrf) and 3D (B, Nt, Nrf) inputs
    if F.dim() == 4:
        K_freq, Bsz, _, _ = F.shape
        F_single = F[0]  # Work with first frequency
        W_single = W[0] if W.dim() == 4 else W
        R_single = R[0] if R.dim() == 4 else R
    else:
        Bsz = F.shape[0]
        F_single = F
        W_single = W
        R_single = R
        K_freq = None
    
    # ---------------------------------------------------
    # 1. Extract active elements
    # ---------------------------------------------------
    f_active = extract_active_elements_pc(F_single, Nt, Nrf)  # (B, Nt, 1)
    
    # ---------------------------------------------------
    # 2. Compute digital covariance V = W W^H (normalized)
    # ---------------------------------------------------
    V = compute_digital_covariance_pc(W_single, Pt, Nrf)  # (B, Nrf, Nrf)
    
    # ---------------------------------------------------
    # 3. Generate Pi matrix: Pi = Σ_rf V_rf * R_block
    # ---------------------------------------------------
    Pi = generate_pi_matrix(V, R_single, Nt, Nrf)  # (B, Nt, Nt)
    
    # ---------------------------------------------------
    # 4. Compute gradient: ∇_f τ = 2 * Pi * f
    #    This is the gradient of f^H Pi f w.r.t. f
    # ---------------------------------------------------
    grad_f_active = 2.0 * torch.bmm(Pi, f_active)  # (B, Nt, 1)
    
    # ---------------------------------------------------
    # 5. Lift gradient from active vector back to F matrix
    # ---------------------------------------------------
    aprf = Nt // Nrf
    eye_nrf = torch.eye(Nrf, device=grad_f_active.device, dtype=grad_f_active.dtype)
    gf = grad_f_active.squeeze(-1).reshape(Bsz, Nrf, aprf)            # (B, Nrf, aprf)
    grad_F_blocks = gf.unsqueeze(-1) * eye_nrf.unsqueeze(0).unsqueeze(2)  # (B, Nrf, aprf, Nrf)
    grad_F = grad_F_blocks.reshape(Bsz, Nt, Nrf)
    
    # ---------------------------------------------------
    # 6. Replicate for all frequencies if needed
    # ---------------------------------------------------
    if K_freq is not None:
        grad_F_all_freq = torch.cat([grad_F.unsqueeze(0)] * K_freq, dim=0)
        return grad_F_all_freq
    else:
        return grad_F


def get_grad_F_com_AP(f_active, H, F, W):
    """
    Computes ∇_a R for partially connected architecture.
    
    Parameters
    ----------
    H : (K, B, M, Nt) channel
    F : (K, B, Nt, Nrf) analog precoder
    W : (K, B, Nrf, M) digital precoder
    
    Returns
    -------
    grad_F : (K, B, Nt, Nrf) gradient of rate w.r.t. F
    """
    
    # Get dimensions from F
    K_freq, Bsz, Nt, Nrf = F.shape
    _, _, M, _ = H.shape
    
    # ---------------------------------------------------
    # 1. Extract active elements: f = N[vec(F)]
    # ---------------------------------------------------
    # f_active shape: (B, Nt, 1)
    
    # ---------------------------------------------------
    # 2. Channel outer products: H̃_m = h_m h_m^H
    # ---------------------------------------------------
    # H shape: (K, B, M, Nt), we need (B, M, Nt, Nt)
    H_single_freq = H[0]  # (B, M, Nt) - work with first frequency for gradient
    H_tilde = (
        H_single_freq.unsqueeze(-1) @ H_single_freq.conj().unsqueeze(-2)
    )  # (B, M, Nt, Nt)
    
    # ---------------------------------------------------
    # 3. Digital covariance terms
    # ---------------------------------------------------
    W_single_freq = W[0]  # (B, Nrf, M)
    w = W_single_freq.permute(0, 2, 1).unsqueeze(-1)   # (B, M, Nrf, 1)
    BmT = w @ w.conj().transpose(-1, -2)  # (B, M, Nrf, Nrf)
    B_all = BmT.sum(dim=1, keepdim=True)  # (B, 1, Nrf, Nrf)
    BmT_tilde = B_all - BmT               # (B, M, Nrf, Nrf)
    
    ln2 = torch.log(torch.tensor(2.0, device=F.device))
    grad_f_active = torch.zeros_like(f_active)
    
    # ---------------------------------------------------
    # 4. Sum over m (users)
    # ---------------------------------------------------
    for m in range(M):
        # Generate Pi matrices using digital covariance and channel outer products
        Pi_mm = generate_pi_matrix(
            BmT[:, m], H_tilde[:, m], Nt, Nrf
        )  # (B, Nt, Nt)
        
        Pi_m_tilde = generate_pi_matrix(
            BmT_tilde[:, m], H_tilde[:, m], Nt, Nrf
        )  # (B, Nt, Nt)
        
        # Quadratic forms
        fH_Pi_mm_f = quad_form(Pi_mm, f_active)  # (B, 1, 1)
        fH_Pi_m_tilde_f = quad_form(Pi_m_tilde, f_active)  # (B, 1, 1)
        
        # Denominators with noise
        denom1 = (
            fH_Pi_mm_f + fH_Pi_m_tilde_f + sigma2
        )
        denom2 = fH_Pi_m_tilde_f + sigma2
        
        # Gradient terms
        term1 = (
            fH_Pi_mm_f / (ln2 * denom1)
        )
        
        term2 = (
            2 * fH_Pi_mm_f
            * torch.bmm(Pi_m_tilde, f_active)
            / (ln2 * denom1 * denom2)
        )
        
        grad_f_active += term1 * f_active - term2
    
    # ---------------------------------------------------
    # 5. Lift gradient from active vector back to F matrix
    # ---------------------------------------------------
    aprf = Nt // Nrf
    eye_nrf = torch.eye(Nrf, device=grad_f_active.device, dtype=grad_f_active.dtype)
    gf = grad_f_active.squeeze(-1).reshape(Bsz, Nrf, aprf)            # (B, Nrf, aprf)
    grad_F_blocks = gf.unsqueeze(-1) * eye_nrf.unsqueeze(0).unsqueeze(2)  # (B, Nrf, aprf, Nrf)
    grad_F = grad_F_blocks.reshape(Bsz, Nt, Nrf)
    
    # Replicate for all frequencies
    grad_F_all_freq = torch.cat([grad_F.unsqueeze(0)] * K_freq, dim=0)
    
    return grad_F_all_freq


# /////////////////////////////////////////////////////////////////////////////////////////
#                             RADAR GRADIENTS
# /////////////////////////////////////////////////////////////////////////////////////////

# ==================================== gradient of tau w.r.t. F ===========================
def get_grad_F_rad(F, W, R):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    if normalize_tau == 1:
        grad_F_K = 2 * (F @ W @ W_H @ F_H - R) @ F @ W @ W_H / torch.linalg.matrix_norm(R[:, 0, :, :], ord='fro') ** 2
    else:
        grad_F_K = 2 * (F @ W @ W_H @ F_H - R) @ F @ W @ W_H
    grad_F_sum = sum(grad_F_K)
    grad_F = grad_F_sum / K
    return grad_F

# ==================================== gradient of tau w.r.t. W ===========================
def get_grad_W_rad(F, W, R):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    if normalize_tau == 1:
        grad_W = 2 * F_H @ (F @ W @ W_H @ F_H - R) @ F @ W / torch.linalg.matrix_norm(R[:, 0, :, :], ord='fro') ** 2
    else:
        grad_W = 2 * F_H @ (F @ W @ W_H @ F_H - R) @ F @ W
    grad_W = grad_W / K
    return grad_W

# ==================compute los based on (15)
def get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, Pt):
    sum_rate = get_sum_rate(H, F, W, Pt)
    # loss = -(sum_rate - OMEGA * sum_error)# / (K * batch_size)

    # For CRB-based loss, we can directly use the CRB value as the loss, since we want to minimize it.
    crb = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt)
    mean_crb = torch.mean(crb)
    loss = -(OMEGA * sum_rate + mean_crb)
    return loss


# ================== compute CRLB gradients =========================
def get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv):

    # reshape A_dot and R_N_inv for batch processing
    A_dot = A_dot.unsqueeze(0).unsqueeze(0) # [1, 1, Nt, Nt]
    R_N_inv = R_N_inv.unsqueeze(0).unsqueeze(0) # [1, 1, Nr, Nr]

    A_dot_H = A_dot.conj().transpose(-2, -1)
    W_H = W.conj().transpose(-2, -1)
    F_H = F.conj().transpose(-2, -1)
    
    M = A_dot_H @ R_N_inv @ A_dot

    inner_mat = W_H @ F_H @ M @ F @ W
    batch_trace = (torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1))
    
    numerator = M @ F @ W @ W_H
    denominator = batch_trace.view(1, -1, 1, 1)
    
    grad_F_crb = numerator / denominator
    
    return grad_F_crb

def get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv):

    A_dot = A_dot.unsqueeze(0).unsqueeze(0) # [1, 1, Nt, Nt]
    R_N_inv = R_N_inv.unsqueeze(0).unsqueeze(0) # [1, 1, Nr, Nr]

    A_dot_H = A_dot.conj().transpose(-2, -1)
    W_H = W.conj().transpose(-2, -1)
    F_H = F.conj().transpose(-2, -1)


    M = A_dot_H @ R_N_inv @ A_dot
    inner_mat = W_H @ F_H @ M @ F @ W
    batch_trace = (torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1))
    
    numerator = F_H @ M @ F @ W
    denominator = batch_trace.view(1, -1, 1, 1)
    grad_W_crb = numerator / denominator
    return grad_W_crb
