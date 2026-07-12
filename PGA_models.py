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

    def __init__(self, step_size, alpha=0.01):
        super().__init__()

        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)
        self.inner_iter_history = []

        # Adaptive scheduling hyperparameter
        self.alpha = alpha

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt,
                    n_iter_outer, n_iter_inner, track_metrics=True):

        rate_init, F, W = initialize(H, Pt, initial_normalization)

        B = len(H[0])

        # Shape: (n_outer, J+1, B)
        # [:, 0:J, :] = metrics after inner F-updates
        # [:, -1, :]  = metrics after W-update
        rate_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt):

            for jj in range(n_inner):

                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

                if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                    print('Error NaN gradients!!!!!!!!!!!!!!!')

                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb

                F = ( F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB )

                F = normalize_power(F, W, H, Pt)

            return F

        inner_iter_history = []
        gradient_norm_history = []
        gradient_norm_history_W = []
        # print(f'Number of inner iterations: {self.step_size.shape[0]}')
        for ii in range(n_iter_outer):

            # ----------------------------------------------------
            # Gradient-norm-based adaptive inner iterations
            # ----------------------------------------------------
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
            grad_J_com = grad_F_com * WEIGHT_F_COM + grad_F_crb * WEIGHT_F_CRB

            if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                print('Error NaN gradients before adaptive J!!!!!!!!!!!!!!!')

            # Adaptive inner iteration count
            # n_inner = _n_inner(ii, n_iter_outer)
            n_inner = self.step_size.shape[0]
    
            if track_metrics:

                inner_iter_history.append(n_inner)
                # Average entry-wise magnitude of ∇_F J
                g_F = torch.abs(grad_J_com).reshape(grad_J_com.shape[0], -1).mean(dim=1)
                gradient_norm_history.append(g_F.mean().item())
                # gradient_norm_history_W.append(torch.linalg.norm(grad_W_k_com.reshape(grad_W_k_com.shape[0], -1), dim=1).mean().item())
                # Run inner loop without checkpoint so that metrics
                # can be recorded after each active inner update.
                for jj in range(n_inner):

                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

                    if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                        print('Error NaN gradients during inner update!!!!!!!!!!!!!!!')

                    delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                    delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb

                    F = ( F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB)

                    # Scale F only, consistent with training path
                    F = normalize_power(F, W, H, Pt)

                    rate_over_iters[ii, jj] = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()

            else:

                F = checkpoint(inner_f_update,F,W,H,xi_0,A_dot,R_N_inv, n_inner, Pt, use_reentrant=False)

            # Projection of analog precoder
            F = project_unit_modulus(F)

            # ----------------------------------------------------
            # Digital precoder update
            # ----------------------------------------------------
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            grad_J_w = grad_W_k_com * WEIGHT_W_COM + grad_W_k_crb * WEIGHT_W_CRB
            
            # Average entry-wise magnitude of ∇_W J
            g_W = torch.abs(grad_J_w).reshape(grad_J_w.shape[0], -1).mean(dim=1)
            gradient_norm_history_W.append(g_W.mean().item())

            W_new = (
                W
                + self.step_size[0][ii][1] * grad_W_k_com * WEIGHT_W_COM
                + self.step_size[0][ii][1] * grad_W_k_crb * WEIGHT_W_CRB
            )

            # Projection / normalization
            F, W = normalize(F, W_new, H, Pt)

            # Record metrics after W-update
            if track_metrics:

                rate_over_iters[ii, -1] = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, -1] = get_crb_fe(
                    H, F, W, xi_0, A_dot, R_N_inv, Pt
                ).detach()
                power_over_iters[ii, -1] = get_power(F, W).detach()

        # --------------------------------------------------------
        # Collect variable-length metric history
        # --------------------------------------------------------
        if track_metrics:

            rate_slots = []
            crb_slots = []
            power_slots = []

            for ii, n_inner_ii in enumerate(inner_iter_history):

                if n_inner_ii > 0:
                    rate_slots.append(rate_over_iters[ii, :n_inner_ii])
                    crb_slots.append(crb_over_iters[ii, :n_inner_ii])
                    power_slots.append(power_over_iters[ii, :n_inner_ii])

                # Add metric after W-update
                rate_slots.append(rate_over_iters[ii, -1:].clone())
                crb_slots.append(crb_over_iters[ii, -1:].clone())
                power_slots.append(power_over_iters[ii, -1:].clone())

            rates = torch.cat(rate_slots, dim=0).detach()
            crb_fes = torch.cat(crb_slots, dim=0).detach()
            power_fes = torch.cat(power_slots, dim=0).detach()

        else:

            # No per-inner metrics are tracked on this path,
            # so retain the fixed rectangular layout.
            rates = rate_over_iters.reshape(
                n_iter_outer * (n_iter_inner + 1), B
            ).detach()

            crb_fes = crb_over_iters.reshape(
                n_iter_outer * (n_iter_inner + 1), B
            ).detach()

            power_fes = power_over_iters.reshape(
                n_iter_outer * (n_iter_inner + 1), B
            ).detach()

        self.inner_iter_history = list(inner_iter_history)
        # print("Adaptive inner iterations:", inner_iter_history)
        # print("Average inner iterations:", sum(inner_iter_history) / len(inner_iter_history))

        return (rates.transpose(0, 1),crb_fes.transpose(0, 1),power_fes.transpose(0, 1),F,W,gradient_norm_history, gradient_norm_history_W)

# ============================================== Unfolded PGA with decaying inner iterations ==============================
class PGA_Unfold_JX_decay(nn.Module):

    def __init__(self, step_size, alpha=0.04, eps=1e-12, J_min=2):
        super().__init__()

        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)
        self.eps = eps
        self.J_min = J_min  # minimum inner iterations to ensure some optimization progress
        self.inner_iter_history = []

        # Adaptive scheduling hyperparameter
        self.alpha = alpha

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt,
                    n_iter_outer, n_iter_inner, track_metrics=True):

        rate_init, F, W = initialize(H, Pt, initial_normalization)

        B = len(H[0])

        # Shape: (n_outer, J+1, B)
        # [:, 0:J, :] = metrics after inner F-updates
        # [:, -1, :]  = metrics after W-update
        rate_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        def _n_inner_from_grad(grad_F_J):

            J_max = self.step_size.shape[0]
            J_min = min(self.J_min, J_max)

            Nt = F.shape[-2]
            Nrf = F.shape[-1]

            g_i = torch.linalg.norm(grad_F_J.reshape(grad_F_J.shape[0], -1), dim=1) / (torch.sqrt(torch.tensor(Nt * Nrf, device=grad_F_J.device, dtype=grad_F_J.real.dtype)) + self.eps)
            g_i = torch.mean(g_i)
            r_i = g_i / (g_i + self.alpha)

            n_inner = int(torch.ceil(J_max * r_i).item())

            n_inner = max(J_min, min(J_max, n_inner))

            return n_inner

        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt):

            for jj in range(n_inner):

                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

                if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                    print('Error NaN gradients!!!!!!!!!!!!!!!')

                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb

                F = (F+ delta_F_com * WEIGHT_F_COM+ delta_F_crb * WEIGHT_F_CRB)
                F = normalize_power(F, W, H, Pt)

            return F

        inner_iter_history = []
        gradient_norm_history = []
        for ii in range(n_iter_outer):

            # ----------------------------------------------------
            # Gradient-norm-based adaptive inner iterations
            # ----------------------------------------------------
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

            if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                print('Error NaN gradients before adaptive J!!!!!!!!!!!!!!!')

            # Adaptive inner iteration count
            grad_F_J = WEIGHT_F_COM * grad_F_com + WEIGHT_F_CRB * grad_F_crb
            n_inner = _n_inner_from_grad(grad_F_J)
            # n_inner = self.step_size.shape[0]
            if track_metrics:

                inner_iter_history.append(n_inner)
                gradient_norm_history.append(torch.linalg.norm(grad_F_J.reshape(grad_F_J.shape[0], -1), dim=1).mean().item())

                # Run inner loop without checkpoint so that metrics
                # can be recorded after each active inner update.
                for jj in range(n_inner):

                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

                    if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                        print('Error NaN gradients during inner update!!!!!!!!!!!!!!!')

                    delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                    delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb

                    F = (F+ delta_F_com * WEIGHT_F_COM+ delta_F_crb * WEIGHT_F_CRB)

                    # Scale F only, consistent with training path
                    F = normalize_power(F, W, H, Pt)

                    rate_over_iters[ii, jj] = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()

            else:

                F = checkpoint(inner_f_update,F,W,H,xi_0,A_dot,R_N_inv,n_inner,Pt,use_reentrant=False)

            # Projection of analog precoder
            F = project_unit_modulus(F)

            # ----------------------------------------------------
            # Digital precoder update
            # ----------------------------------------------------
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)

            W_new = (W+ self.step_size[0][ii][1] * grad_W_k_com * WEIGHT_W_COM+ self.step_size[0][ii][1] * grad_W_k_crb * WEIGHT_W_CRB)

            # Projection / normalization
            F, W = normalize(F, W_new, H, Pt)

            # Record metrics after W-update
            if track_metrics:

                rate_over_iters[ii, -1] = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, -1] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                power_over_iters[ii, -1] = get_power(F, W).detach()

        # --------------------------------------------------------
        # Collect variable-length metric history
        # --------------------------------------------------------
        if track_metrics:

            rate_slots = []
            crb_slots = []
            power_slots = []

            for ii, n_inner_ii in enumerate(inner_iter_history):

                if n_inner_ii > 0:
                    rate_slots.append(rate_over_iters[ii, :n_inner_ii])
                    crb_slots.append(crb_over_iters[ii, :n_inner_ii])
                    power_slots.append(power_over_iters[ii, :n_inner_ii])

                # Add metric after W-update
                rate_slots.append(rate_over_iters[ii, -1:].clone())
                crb_slots.append(crb_over_iters[ii, -1:].clone())
                power_slots.append(power_over_iters[ii, -1:].clone())

            rates = torch.cat(rate_slots, dim=0).detach()
            crb_fes = torch.cat(crb_slots, dim=0).detach()
            power_fes = torch.cat(power_slots, dim=0).detach()

        else:

            # No per-inner metrics are tracked on this path,
            # so retain the fixed rectangular layout.
            rates = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
            crb_fes = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
            power_fes = power_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()

        self.inner_iter_history = list(inner_iter_history)
        # print("Adaptive inner iterations:", inner_iter_history)
        # print("Average inner iterations:", sum(inner_iter_history) / len(inner_iter_history))
        # print("Gradient norms at outer iterations:", gradient_norm_history)

        return (rates.transpose(0, 1),crb_fes.transpose(0, 1),power_fes.transpose(0, 1),F,W, gradient_norm_history) 


class PGA_Unfold_JX_partial(nn.Module):
    def __init__(self, step_size, Nt=None, Nrf=None, mask=None, alpha=0.01):
        super().__init__()

        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)
        self.inner_iter_history = []
        self.alpha = alpha

        if mask is not None:
            self.register_buffer('mask', mask.float())
        elif Nt is not None and Nrf is not None:
            assert Nt % Nrf == 0, "Number of antennas (Nt) must be perfectly divisible by RF chains (Nrf) for symmetric sub-connection."
            ant_per_rf = Nt // Nrf
            template_mask = torch.zeros(Nt, Nrf)
            for r in range(Nrf):
                template_mask[r * ant_per_rf : (r + 1) * ant_per_rf, r] = 1.0
            self.register_buffer('mask', template_mask)
        else:
            raise ValueError("You must provide either a explicit 'mask' tensor or both 'Nt' and 'Nrf' dimensions.")

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner, track_metrics=True):

        _, F, W = initialize(H, Pt, initial_normalization)
        
        # 2. Apply Mask immediately after initialization to clear unauthorized paths
        F = F * self.mask.to(F.device)
        
        B = len(H[0])

        rate_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        ## Inner loop
        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt):

            for jj in range(n_inner):

                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                    print('Error NaN gradients!!!!!!!!!!!!!!!')

                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb
 
                # 2. Mask applied during gradient update step
                F = ( F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB ) * self.mask.to(F.device)

                F = normalize_power(F, W, H, Pt)
                F = F * self.mask.to(F.device) # Ensure zero-mask is perfectly maintained after power scaling

            return F

        gradient_norm_history, gradient_norm_history_W = [], []

        for ii in range(n_iter_outer):

            n_inner = self.step_size.shape[0]
    
            if track_metrics:
                for jj in range(n_inner):

                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

                    if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                        print('Error NaN gradients during inner update!!!!!!!!!!!!!!!')

                    delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                    delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb

                    # 2. Mask applied during tracked gradient update step
                    F = ( F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB) * self.mask.to(F.device)

                    # Scale F only, consistent with training path
                    F = normalize_power(F, W, H, Pt)
                    F = F * self.mask.to(F.device)

                    rate_over_iters[ii, jj] = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()

            else:

                F = checkpoint(inner_f_update, F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt, use_reentrant=False)

            F = project_unit_modulus(F) * self.mask.to(F.device)

            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            grad_J_w = grad_W_k_com * WEIGHT_W_COM + grad_W_k_crb * WEIGHT_W_CRB
            
            # Average entry-wise magnitude of ∇_W J
            g_W = torch.abs(grad_J_w).reshape(grad_J_w.shape[0], -1).mean(dim=1)
            gradient_norm_history_W.append(g_W.mean().item())

            W_new = W + self.step_size[0][ii][1] * grad_W_k_com * WEIGHT_W_COM + self.step_size[0][ii][1] * grad_W_k_crb * WEIGHT_W_CRB

            # Projection / normalization
            F, W = normalize(F, W_new, H, Pt)
            F = F * self.mask.to(F.device) # Safeguard mask safety after composite normalization

            # Record metrics after W-update
            if track_metrics:

                rate_over_iters[ii, -1] = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, -1] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                power_over_iters[ii, -1] = get_power(F, W).detach()


        print(f'F matrix after {n_iter_outer} outer iterations:\n{F}')

        rates = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        crb_fes = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        power_fes = power_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()

        return (rates.transpose(0, 1),crb_fes.transpose(0, 1),power_fes.transpose(0, 1),F,W,gradient_norm_history,gradient_norm_history_W,)
    
class PGA_Unfold_JX_partial_decay(nn.Module):
    def __init__(self, step_size=None, Nt=None, Nrf=None, alpha=0.04, eps=1e-12, J_min=2):
        super().__init__()

        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)
        self.eps = eps
        self.J_min = J_min
        self.inner_iter_history = []

        # Adaptive scheduling hyperparameter
        self.alpha = alpha

        if Nt is not None and Nrf is not None:
            assert Nt % Nrf == 0, "Number of antennas (Nt) must be perfectly divisible by RF chains (Nrf) for symmetric sub-connection."
            ant_per_rf = Nt // Nrf
            template_mask = torch.zeros(Nt, Nrf)
            for r in range(Nrf):
                template_mask[r * ant_per_rf : (r + 1) * ant_per_rf, r] = 1.0
            self.register_buffer('mask', template_mask)
        else:
            raise ValueError("You must provide either a explicit 'mask' tensor or both 'Nt' and 'Nrf' dimensions.")

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner, track_metrics=True):

        _, F, W = initialize(H, Pt, initial_normalization)
        F = F * self.mask.to(F.device)
        
        B = len(H[0])

        rate_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        crb_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)
        power_over_iters = torch.zeros(n_iter_outer, n_iter_inner + 1, B, device=H.device)

        ## Inner loop
        def _n_inner_from_grad(grad_F_J):

            J_max = self.step_size.shape[0]
            J_min = min(self.J_min, J_max)

            Nt = F.shape[-2]
            Nrf = F.shape[-1]

            g_i = torch.linalg.norm(grad_F_J.reshape(grad_F_J.shape[0], -1), dim=1) / (torch.sqrt(torch.tensor(Nt * Nrf, device=grad_F_J.device, dtype=grad_F_J.real.dtype)) + self.eps)
            g_i = torch.mean(g_i)
            r_i = g_i / (g_i + self.alpha)

            n_inner = int(torch.ceil(J_max * r_i).item())

            n_inner = max(J_min, min(J_max, n_inner))

            return n_inner
        
        def inner_f_update(F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt):

            for jj in range(n_inner):

                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

                if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                    print('Error NaN gradients!!!!!!!!!!!!!!!')

                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb
 
                # 2. Mask applied during gradient update step
                F = ( F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB ) * self.mask.to(F.device)

                F = normalize_power(F, W, H, Pt)
                F = F * self.mask.to(F.device) # Ensure zero-mask is perfectly maintained after power scaling

            return F

        gradient_norm_history, gradient_norm_history_W = [], []

        for ii in range(n_iter_outer):

            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
            # calculate the grad_F_J
            grad_F_J = WEIGHT_F_COM * grad_F_com + WEIGHT_F_CRB * grad_F_crb
            n_inner = _n_inner_from_grad(grad_F_J)
    
            if track_metrics:
                for jj in range(n_inner):

                    grad_F_com = get_grad_F_com(H, F, W)
                    grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)

                    if grad_F_com.isnan().any() or grad_F_crb.isnan().any():
                        print('Error NaN gradients during inner update!!!!!!!!!!!!!!!')

                    delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                    delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb

                    # 2. Mask applied during tracked gradient update step
                    F = ( F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB) * self.mask.to(F.device)

                    # Scale F only, consistent with training path
                    F = normalize_power(F, W, H, Pt)
                    F = F * self.mask.to(F.device)

                    rate_over_iters[ii, jj] = get_sum_rate(H, F, W, Pt).detach()
                    crb_over_iters[ii, jj] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                    power_over_iters[ii, jj] = get_power(F, W).detach()

            else:

                F = checkpoint(inner_f_update, F, W, H, xi_0, A_dot, R_N_inv, n_inner, Pt, use_reentrant=False)

            F = project_unit_modulus(F) * self.mask.to(F.device)

            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            grad_J_w = grad_W_k_com * WEIGHT_W_COM + grad_W_k_crb * WEIGHT_W_CRB
            
            # Average entry-wise magnitude of ∇_W J
            g_W = torch.abs(grad_J_w).reshape(grad_J_w.shape[0], -1).mean(dim=1)
            gradient_norm_history_W.append(g_W.mean().item())

            W_new = W + self.step_size[0][ii][1] * grad_W_k_com * WEIGHT_W_COM + self.step_size[0][ii][1] * grad_W_k_crb * WEIGHT_W_CRB

            # Projection / normalization
            F, W = normalize(F, W_new, H, Pt)
            F = F * self.mask.to(F.device) # Safeguard mask safety after composite normalization

            # Record metrics after W-update
            if track_metrics:

                rate_over_iters[ii, -1] = get_sum_rate(H, F, W, Pt).detach()
                crb_over_iters[ii, -1] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt).detach()
                power_over_iters[ii, -1] = get_power(F, W).detach()


        # print(f'F matrix after {n_iter_outer} outer iterations:\n{F}')

        rates = rate_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        crb_fes = crb_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()
        power_fes = power_over_iters.reshape(n_iter_outer * (n_iter_inner + 1), B).detach()

        return (rates.transpose(0, 1),crb_fes.transpose(0, 1),power_fes.transpose(0, 1),F,W,gradient_norm_history,gradient_norm_history_W,)
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

# ================== Compute exponentially weighted deep-supervision loss
def get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, Pt, beta=0.97):

    sum_rate = get_sum_rate(H, F, W, Pt)
    crb = get_crb_fe(H, F, W,xi_0, A_dot, R_N_inv, Pt)

    mean_crb = torch.mean(crb)

    loss = -(OMEGA * sum_rate + mean_crb)
    # loss = -( sum_rate + OMEGA * mean_crb)

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
