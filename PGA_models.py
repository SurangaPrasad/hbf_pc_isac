import torch
import torch.nn as nn
from utility import *
from utility import _get_real_dtype_like

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
        device = H.device
        real_dtype = _get_real_dtype_like(H)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)  # save beampattern errors over iterations
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
    def execute_PGA(self, H, R, Pt, n_iter_outer):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        device = H.device
        real_dtype = _get_real_dtype_like(H)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)  # save beampattern errors over iterations
        # update F and W over iterations
        for ii in range(n_iter_outer):
            # update F
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_rad = get_grad_F_rad(F, W, R)
            # self.step_size[ii][0]
            delta_F_com = self.step_size[ii][0] * grad_F_com
            delta_F_rad = self.step_size[ii][0] * grad_F_rad
            F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD

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

# ============================================== Proposed PGA model light=============================
class PGA_Unfold_J10(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        device = H.device
        real_dtype = _get_real_dtype_like(H)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)# save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)# save beam errors over iterations

        for ii in range(n_iter_outer):
            # update F over
            for jj in range(n_iter_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                if grad_F_com.isnan().any() or grad_F_rad.isnan().any(): # check gradient
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                # normalize by power to ensure non-NaN gradients if F becomes too large
                #if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                F = normalize_power(F, W, H, Pt)
            # Projection
            F = project_unit_modulus(F)

            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k + 1] * grad_W_k_com[k]
                delta_W_rad = self.step_size[0][ii][k + 1] * grad_W_k_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD
            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            # print(rate_over_iters[ii])
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)

        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W
# ============================================== Proposed PGA model light for PC======================
class PGA_Unfold_J10_PC(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization, pc=True)
        device = H.device
        real_dtype = _get_real_dtype_like(H)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)# save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)# save beam errors over iterations

        pc_mask = generage_partial_connection_mask(Nt, Nrf, device=F.device, dtype=F.dtype)
        for ii in range(2):
            # update F over
            for jj in range(2):    
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)

                grad_F_com = grad_F_com * pc_mask
                grad_F_rad = grad_F_rad * pc_mask


                # print(f'F matrix element: {F[0,0, :, :]}')
                if grad_F_com.isnan().any() or grad_F_rad.isnan().any(): # check gradient
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                delta_F_com = clamp_complex_magnitude(delta_F_com, 0.5)
                delta_F_rad = clamp_complex_magnitude(delta_F_rad, 0.5)
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                F = sanitize_complex_tensor(F)
                F = F * pc_mask
                F = sanitize_complex_tensor(F)
                # normalize by power to ensure non-NaN gradients if F becomes too large
                #if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                F = normalize_power(F, W, H, Pt)
                F = sanitize_complex_tensor(F)
            # Projection
            F = project_unit_modulus(F, active_mask=pc_mask)
            F = sanitize_complex_tensor(F)
            
            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k + 1] * grad_W_k_com[k]
                delta_W_rad = self.step_size[0][ii][k + 1] * grad_W_k_rad[k]
                delta_W_com = clamp_complex_magnitude(delta_W_com, 0.5)
                delta_W_rad = clamp_complex_magnitude(delta_W_rad, 0.5)
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD
                W_new[k] = sanitize_complex_tensor(W_new[k])
            
            # Projection
            F, W = normalize(F, W_new, H, Pt)
            F = sanitize_complex_tensor(F)
            W = sanitize_complex_tensor(W)
            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            # print(rate_over_iters[ii])
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)

        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W
# ============================================== Proposed PGA model=============================
class PGA_Unfold_J20(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        device = H.device
        real_dtype = _get_real_dtype_like(H)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)# save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=device, dtype=real_dtype)# save beam errors over iterations

        for ii in range(n_iter_outer):
            # update F over
            for jj in range(n_iter_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                if grad_F_com.isnan().any() or grad_F_rad.isnan().any(): # check gradient
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                # normalize by power to ensure non-NaN gradients if F becomes too large
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            # Projection
            F = project_unit_modulus(F)

            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k + 1] * grad_W_k_com[k]
                delta_W_rad = self.step_size[0][ii][k + 1] * grad_W_k_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD

            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            # print(rate_over_iters[ii])
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)

        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W


# /////////////////////////////////////////////////////////////////////////////////////////
#                             COMM GRADIENTS
# /////////////////////////////////////////////////////////////////////////////////////////


# ==================================== gradient of R_mk w.r.t. F ===========================
def get_grad_F_com(H, F, W, sigma2, K, Nt, Nrf):
    """
    Vectorized GPU-friendly version of get_grad_F_com
    Removes the for-loop over m using batched matrix multiplications
    """

    device = F.device
    dtype = F.dtype

    # Hermitian transposes
    F_H = F.transpose(2, 3).conj()          # [K, train, Nrf, Nt]
    W_H = W.transpose(2, 3).conj()          # [K, train, M, Nrf]

    # V = W W^H
    V = W @ W_H                             # [K, train, Nrf, Nrf]

    # --------------------------------------------------
    # Construct Htilde_mk for ALL m
    # --------------------------------------------------
    # H: [K, train, M, Nt]
    h = H.permute(0, 1, 3, 2)               # [K, train, Nt, M]
    Htilde = h[..., :, None] @ h[..., None, :].conj()
    # Htilde: [K, train, M, Nt, Nt]

    # --------------------------------------------------
    # Construct V_mk = V - w_m w_m^H for ALL m
    # --------------------------------------------------
    w = W.permute(0, 1, 3, 2)               # [K, train, Nrf, M]
    w_outer = w[..., :, None] @ w[..., None, :].conj()
    # w_outer: [K, train, M, Nrf, Nrf]

    V_exp = V[:, :, None, :, :]             # [K, train, 1, Nrf, Nrf]
    V_mk = V_exp - w_outer                  # [K, train, M, Nrf, Nrf]

    # --------------------------------------------------
    # Expand F for broadcasting
    # --------------------------------------------------
    F_exp = F[:, :, None, :, :]             # [K, train, 1, Nt, Nrf]
    F_H_exp = F_H[:, :, None, :, :]         # [K, train, 1, Nrf, Nt]

    log2 = torch.log(torch.tensor(2.0, device=device, dtype=dtype))

    # --------------------------------------------------
    # grad_F_1 (all m)
    # --------------------------------------------------
    C = F_exp @ V_exp @ F_H_exp @ Htilde
    trace_C = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    denom_1 = log2 * (trace_C + sigma2)

    grad_F_1 = (
        Htilde @ F_exp @ V_exp
    ) / (denom_1[..., None, None] + 1e-4)

    # --------------------------------------------------
    # grad_F_2 (all m)
    # --------------------------------------------------
    C1 = F_exp @ V_mk @ F_H_exp @ Htilde
    trace_C1 = torch.diagonal(C1, dim1=-2, dim2=-1).sum(-1)
    denom_2 = log2 * (trace_C1 + sigma2)

    grad_F_2 = (
        Htilde @ F_exp @ V_mk
    ) / (denom_2[..., None, None] + 1e-4)

    # --------------------------------------------------
    # Sum over m (this replaces the for-loop)
    # --------------------------------------------------
    grad_F_sum_M = (grad_F_1 - grad_F_2).sum(dim=2)   # [K, train, Nt, Nrf]

    # --------------------------------------------------
    # Final reduction over K (same as original)
    # --------------------------------------------------
    grad_F_sum_K = grad_F_sum_M.sum(dim=0) / K
    grad_F = grad_F_sum_K.unsqueeze(0).repeat(K, 1, 1, 1)

    return grad_F

def get_grad_W_com(H, F, W, sigma2, K):
    """
    Fully vectorized GPU-friendly version of get_grad_W_com
    Removes the for-loop over m using batched matrix multiplications
    """

    device = W.device
    dtype = W.dtype

    # Dimensions
    K_, train_size, M, Nt = H.shape
    _, _, Nrf, _ = W.shape

    # Hermitian transposes
    F_H = F.transpose(2, 3).conj()      # [K, train, Nrf, Nt]
    W_H = W.transpose(2, 3).conj()      # [K, train, M, Nrf]

    log2 = torch.log(torch.tensor(2.0, device=device, dtype=dtype))

    # --------------------------------------------------
    # Construct Htilde_mk for ALL m
    # --------------------------------------------------
    # H: [K, train, M, Nt]
    h = H.permute(0, 1, 3, 2)            # [K, train, Nt, M]
    Htilde = h[..., :, None] @ h[..., None, :].conj()
    # [K, train, M, Nt, Nt]

    # --------------------------------------------------
    # Hbar_mk = F^H Htilde F   (for all m)
    # --------------------------------------------------
    F_exp = F[:, :, None, :, :]          # [K, train, 1, Nt, Nrf]
    F_H_exp = F_H[:, :, None, :, :]      # [K, train, 1, Nrf, Nt]

    Hbar = F_H_exp @ Htilde @ F_exp
    # [K, train, M, Nrf, Nrf]

    # --------------------------------------------------
    # Precompute W W^H
    # --------------------------------------------------
    V = W @ W_H                          # [K, train, Nrf, Nrf]
    V_exp = V[:, :, None, :, :]          # [K, train, 1, Nrf, Nrf]

    # --------------------------------------------------
    # grad_W_1  (same for all m)
    # --------------------------------------------------
    C1 = V_exp @ Hbar
    trace_1 = torch.diagonal(C1, dim1=-2, dim2=-1).sum(-1)
    denom_1 = log2 * (trace_1 + sigma2)

    grad_W_1 = (
        Hbar @ W[:, :, None, :, :]
    ) / (denom_1[..., None, None] + 1e-4)
    # [K, train, M, Nrf, M]

    # --------------------------------------------------
    # grad_W_2  (same computation, masked per m)
    # --------------------------------------------------
    C2 = (W @ W_H)[:, :, None, :, :] @ Hbar
    trace_2 = torch.diagonal(C2, dim1=-2, dim2=-1).sum(-1)
    denom_2 = log2 * (trace_2 + sigma2)

    grad_W_2 = (
        Hbar @ W[:, :, None, :, :]
    ) / (denom_2[..., None, None] + 1e-4)

    # --------------------------------------------------
    # Mask diagonal (remove column m contribution)
    # --------------------------------------------------
    mask = torch.ones(M, M, device=device, dtype=dtype)
    mask.fill_diagonal_(0.0)
    mask = mask[None, None, None, :, :]      # broadcastable

    grad_W_2_masked = grad_W_2 * mask

    # --------------------------------------------------
    # Sum over m (replaces the loop)
    # --------------------------------------------------
    grad_W = (grad_W_1 - grad_W_2_masked).sum(dim=2)

    # --------------------------------------------------
    # Final normalization
    # --------------------------------------------------
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

# ==================compute los based on (15)
def get_sum_loss(F, W, H, R, Pt, batch_size):
    sum_rate = get_sum_rate(H, F, W, Pt)
    # sum_rate = sum(rate)
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    if normalize_tau == 1:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2 / torch.linalg.matrix_norm(R[:, 0, :, :] , ord='fro') ** 2
    else:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
    # sum_error = sum(sum(error))
    sum_error = torch.mean(error)
    loss = -(sum_rate - OMEGA * sum_error)# / (K * batch_size)
    return loss



