import torch
import torch.nn as nn
from utility import *

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
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]))  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]))  # save beampattern errors over iterations
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
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]))  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]))  # save beampattern errors over iterations
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
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save beam errors over iterations

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
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save beam errors over iterations

        pc_mask = generage_partial_connection_mask(Nt, Nrf).to(device=F.device, dtype=F.dtype)
        for ii in range(n_iter_outer):
            # update F over
            for jj in range(n_iter_inner):    
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


class PGA_Unfold_J10_PC_AP(nn.Module):
    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization, pc=True)
        print(f'The size of F: {F.shape}')
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save beam errors over iterations

        pc_mask = generage_partial_connection_mask(Nt, Nrf).to(device=F.device, dtype=F.dtype)
        for ii in range(1):
            # update F over
            for jj in range(1):    
                grad_F_com_AP = get_grad_F_com_AP(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)

                # print(f'F matrix element: {F[0,0, :, :]}')
                if grad_F_com_AP.isnan().any() or grad_F_rad.isnan().any(): # check gradient
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com_AP
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
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save beam errors over iterations

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
def get_grad_F_com(H, F, W):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()  # _H, _T means hermitian and transpose of a matrix
    V = W @ W_H  # K x train_size x Nrf x Nrf
    grad_F_sum_M = torch.zeros(len(H[0]), Nt, Nrf)
    for m in range(M):
        W_m = W[:, :, :, torch.arange(W.size(3)) != m]
        V_mk = W_m @ torch.transpose(W_m, 2, 3).conj()  # need to change to remove 1 column
        h_mk0 = torch.unsqueeze(H[:, :, m, :], dim=2)
        h_mk = torch.transpose(h_mk0, 2, 3)
        h_mk_H = torch.transpose(h_mk, 2, 3).conj()
        Htilde_mk = h_mk @ h_mk_H

        A = F @ V
        B = A @ F_H
        C = B @ Htilde_mk
        denom_1 = np.log(2) * (get_trace(C) + sigma2)
        grad_F_1 = Htilde_mk @ F @ V / (denom_1[:, :, None, None]+1e-4)  # expand dimension

        A1 = F @ V_mk
        B1 = A1 @ F_H
        C1 = B1 @ Htilde_mk
        denom_2 = np.log(2) * (get_trace(C1) + sigma2)
        grad_F_2 = Htilde_mk @ F @ V_mk / (denom_2[:, :, None, None]+1e-4)  # expand dimension

        grad_F_sum_M = grad_F_sum_M + (grad_F_1 - grad_F_2)

    grad_F_sum_K = sum(grad_F_sum_M) / K 
    grad_F = torch.cat(((grad_F_sum_K[None, :, :, :],) * K), 0)
    return grad_F

def get_grad_W_com(H, F, W):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    V = W @ W_H  # K x train_size x Nrf x Nrf
    grad_W = torch.zeros(len(H), len(H[0]), Nrf, M)

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
        mask_m = torch.ones(len(H), len(H[0]), Nrf, M)
        mask_m[:, :, :, m] = torch.zeros(len(H), len(H[0]), Nrf)
        grad_W_2_masked = grad_W_2 * mask_m  # need element-wise multiplication for masking
        grad_W = grad_W + (grad_W_1 - grad_W_2_masked)

    grad_W = grad_W / K
    return grad_W

def get_grad_F_com_AP(H, F, W):
    a = extract_active_elements(F)  # (B, N, 1)
    print(f'F matrix element: {F[0,0, :, :]}')
    print(f'Active elements: {a[0, :, :]}')
    return a

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



