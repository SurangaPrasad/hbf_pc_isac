import torch
import torch.nn as nn
import sys
import h5py
import scipy.io
from system_config import *
import matplotlib.pyplot as plt


def randn_complex(shape, device=None):
    """Sample a complex tensor with IID unit-variance entries."""
    real = torch.randn(shape, dtype=REAL_DTYPE, device=device)
    imag = torch.randn(shape, dtype=REAL_DTYPE, device=device)
    return torch.complex(real, imag)


# ==================================== initialize F and W ===========================
def initialize(H, Pt, normalization, pc=False):
    if init_scheme == 'conv':
        # randomizing F
        F = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    elif init_scheme == 'prop':  # use Le Liang's paper: Low-Complexity Hybrid Precoding in Massive Multiuser MIMO Systems
        # if K == 1:
        #     F = H / torch.abs(H)
        #     F = torch.transpose(F, 2, 3)
        #     Hp = H.conj()
        #     Q = torch.linalg.pinv(Hp)
        #     FQ = torch.linalg.pinv(F) @ Q
        #     W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))
        # else:
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, len(H[0]), Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        elif Nrf > M:  # more RF chains than user, need sensing channel as well
            # Determine G
            G = get_mat_G(H,K//2,snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, len(H[0]), Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Fpinv = torch.linalg.pinv(F)
                Wtmp = torch.bmm(Fpinv, Xzf)
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        else:
            sys.stderr.write('Error: Wrong RF chain configuration....\n')
        F = F * generage_partial_connection_mask(Nt, Nrf, device=F.device) if pc else F
    elif init_scheme == 'svd':
        U, S, V_H = torch.linalg.svd(H)
        V = V_H
        # V = torch.transpose(V_H, 2, 3).conj()
        F = V[:, :, :, :Nrf]
        F = F / torch.abs(F)
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))
    else:
        R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
        at = at0[:, : batch_size, :, :]
        angles_theta = np.around(theta[0, :] * 180 / np.pi)
        idx_snr = np.where(angles_theta == 0)
        at_tmp = at[0, 0, :, idx_snr]
        at1 = at_tmp[:, 0, 0]
        F = H / torch.abs(H)
        F = torch.transpose(F, 2, 3)
        F[:, :, :, 0] = at1
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))

    # rate_0 = get_sum_rate(H, F, W)
    # print(rate_0)
    if normalization == 1:
        # normalize both F and W
        F, W = normalize(F, W, H, Pt)
    else:
        # only normalize W for power constraint
        B = len(H[0])
        norm2_FW = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)  # (B,)
        if torch.is_tensor(Pt) and Pt.dim() >= 1:
            Pt_vec = Pt.to(dtype=norm2_FW.dtype, device=F.device)
        else:
            Pt_vec = torch.full((B,), float(Pt), dtype=norm2_FW.dtype, device=F.device)
        W = torch.sqrt(Pt_vec / norm2_FW).view(B, 1, 1) * W
    # rate_0 = get_sum_rate(H, F, W)
    rate_init = torch.zeros(1, len(H[0]), device=H.device)
    # beam_error_init = torch.zeros(1, len(H[0]))
    rate_init[0, :] = get_sum_rate(H, F, W, Pt)
    # beam_error_init[0, :] = get_beam_error(H, F, W, R, Pt)

    return rate_init, F, W


# ==================================== initialize F and W with different methods for comparison ===========================
def initialize_schemes(H, R, Pt, init_method):
    if init_method == 'conv':
        # randomizing F
        F = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    elif init_method == 'prop':  # use Le Liang's paper: Low-Complexity Hybrid Precoding in Massive Multiuser MIMO Systems
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        elif Nrf > M:  # more RF chains than user, need sensing channel as well
            # Determine G
            G = get_mat_G(H, K // 2, snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
    elif init_method == 'svd':
        if Nrf == M:
            U, S, V_H = torch.linalg.svd(H)
            V = V_H
            # V = torch.transpose(V_H, 2, 3).conj()
            F = V[:, :, :, :Nrf]
            F = F / torch.abs(F)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k, :, :, :]
                Hp = Hk.conj()
                Q = torch.linalg.pinv(Hp)
                FQ = torch.linalg.pinv(F) @ Q
                fro_norm = torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = FQ / fro_norm
        elif Nrf > M:
            # Determine G
            G = get_mat_G_SVD(H, K // 2, snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Fpinv = torch.linalg.pinv(F)
                Wtmp = torch.bmm(Fpinv, Xzf)
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)

    else:
        R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
        at = at0[:, : batch_size, :, :]
        angles_theta = np.around(theta[0, :] * 180 / np.pi)
        idx_snr = np.where(angles_theta == 0)
        at_tmp = at[0, 0, :, idx_snr]
        at1 = at_tmp[:, 0, 0]
        F = H / torch.abs(H)
        F = torch.transpose(F, 2, 3)
        F[:, :, :, 0] = at1
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))

    # only normalize W for power constraint
    B = len(H[0])
    norm2_FW = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
    if torch.is_tensor(Pt) and Pt.dim() >= 1:
        Pt_vec = Pt.to(dtype=norm2_FW.dtype, device=F.device)
    else:
        Pt_vec = torch.full((B,), float(Pt), dtype=norm2_FW.dtype, device=F.device)
    W = torch.sqrt(Pt_vec / norm2_FW).view(B, 1, 1) * W

    # rate_0 = get_sum_rate(H, F, W)
    rate_init = torch.zeros(1, len(H[0]), device=H.device)
    beam_error_init = torch.zeros(1, len(H[0]), device=H.device)
    rate_init[0, :] = get_sum_rate(H, F, W, Pt)
    beam_error_init[0, :] = get_beam_error(H, F, W, R, Pt)

    return rate_init, beam_error_init, F, W


# ================== get matrix G for initalization when Nrf > K
def get_mat_G(H,fre_indx,snr_dB):
    G = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
    Htmp = torch.transpose(H[fre_indx, :, :, :], 1, 2)
    G[:, :, :M] = Htmp

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch = at0[:, : batch_size, :, :]
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp = at_batch[0, :, :, angle_index]
        at = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G

def get_mat_G_SVD(H,fre_indx,snr_dB):
    G = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
    U, S, V_H = torch.linalg.svd(H)
    V = V_H
    G[:, :, :M] = V[:, :, :, :M]

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch = at0[:, : batch_size, :, :]
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp = at_batch[0, :, :, angle_index]
        at = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G
# ==================================== compute sum rate of MU-MISO system for each subcarrier ===========================
def get_sum_rate(H, F, W, Pt, skip_unit_modulus=False):

    # Normalize
    F, W = normalize(F, W, H, Pt, skip_unit_modulus=skip_unit_modulus)

    # ================= Power constraint check =================
    power_high_threshold = Pt + 1e-3

    power = torch.linalg.matrix_norm(F @ W, dim=(-2, -1)) ** 2  # (K, B)
    sum_power = torch.mean(power)

    # if torch.any(sum_power > power_high_threshold):
        # sys.stderr.write('Error: power constraint violated\n')

    # ================= Precompute =================
    F_H = F.conj().transpose(-2, -1)             # (K, B, Nrf, Nt)
    W_H = W.conj().transpose(-2, -1)             # (K, B, M, Nrf)

    V = W @ W_H                                  # (K, B, Nrf, Nrf)

    # ================= Build V_m =================
    Mval = W.shape[-1]

    mask = (1 - torch.eye(Mval, device=W.device, dtype=W.dtype))  # (M, M)

    # Apply mask: zero each column m
    W_m_all = W.unsqueeze(2) * mask.view(1, 1, Mval, 1, Mval)
    # shape: (K, B, M, Nrf, M)

    V_m_all = W_m_all @ W_m_all.conj().transpose(-1, -2)
    # (K, B, M, Nrf, Nrf)

    # ================= Channel outer products =================
    # H: (K, B, M, Nt)
    h = H.unsqueeze(-1)                          # (K, B, M, Nt, 1)
    Htilde = h @ h.conj().transpose(-1, -2)      # (K, B, M, Nt, Nt)

    # ================= Trace 1 =================
    FVF_H = F @ V @ F_H                          # (K, B, Nt, Nt)

    trace_1 = (
        (FVF_H.unsqueeze(2) @ Htilde)
        .diagonal(dim1=-1, dim2=-2)
        .sum(-1)
    )  # (K, B, M)

    # ================= Trace 2 =================
    FVmFH = F.unsqueeze(2) @ V_m_all @ F_H.unsqueeze(2)

    trace_2 = (
        (FVmFH @ Htilde)
        .diagonal(dim1=-1, dim2=-2)
        .sum(-1)
    )  # (K, B, M)

    # ================= Rate =================
    rate = (
        torch.log2(trace_1 + sigma2)
        - torch.log2(trace_2 + sigma2)
    ).real.sum(dim=-1)  # (K, B)

    sum_rate = torch.mean(rate, dim=0)   # mean over K -> (B,)

    return sum_rate


# ==================================== compute tau function ===========================
def get_beam_error(H, F, W, R, Pt):
    F, W = normalize(F, W, H, Pt)
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    if normalize_tau == 1:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2 / torch.linalg.matrix_norm(R, ord='fro') ** 2
    else:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
    sum_error = torch.mean(error)
    return sum_error

# ==================================== compute CRB  fishery equation function ===========================
def get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt, skip_unit_modulus=False):
    F, W = normalize(F, W, H, Pt, skip_unit_modulus=skip_unit_modulus)
    
    A_dot = A_dot.unsqueeze(0).unsqueeze(0) # [1, 1, Nt, Nt]
    R_N_inv = R_N_inv.unsqueeze(0).unsqueeze(0) # [1, 1, Nr, Nr]

    A_dot_H = A_dot.conj().transpose(-2, -1)
    W_H = W.conj().transpose(-2, -1)
    F_H = F.conj().transpose(-2, -1)
    
    M = A_dot_H @ R_N_inv @ A_dot
    inner_mat = W_H @ F_H @ M @ F @ W
    batch_trace = (torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1))  # [K, batch_size]

    fim = batch_trace.sum(0).real  # sum over K users -> [batch_size]

    crb = torch.log(fim) + torch.log(2*torch.tensor(xi_0) ** 2)  # [batch_size]

    # crb_real = 1 / (2 * (xi_0 ** 2) * fim)
    # print(f"CRB (real): {crb_real}")

    return crb

# ====================================compute power of F and W ==========================

def get_power(F, W):
    # power = frobinus_norm(F * W)^2
    power = torch.linalg.matrix_norm(F @ W, ord='fro') ** 2
    return power
# ==================================== compute MSE ===========================
def get_MSE(F, W, at, R, Pt):
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    at_H = torch.transpose(at, 2, 3).conj()
    beampattern = torch.real(torch.diagonal(at_H @ X @ X_H @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_mean = torch.mean(beampattern,0)
    # benchmark beampatter
    beam_bm = torch.real(torch.diagonal(at_H @ R @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_bm_mean = torch.mean(beam_bm,0)

    MSE = (torch.abs(beam_bm_mean - beam_mean)) ** 2
    MSE_mean = 10 * torch.log10(torch.mean(torch.mean(MSE, 1)))  # average over channel and get sum
    return MSE_mean


# ==================================== compute trace of matrix A ===========================
def get_trace(A):
    trace_A = torch.diagonal(A, offset=0, dim1=-1, dim2=-2).sum(-1)  # sum all diagonal elements
    return trace_A


# ======== normalization to meet constant modulus and power constraint ===========================
def normalize(F, W, H, Pt, skip_unit_modulus=False):
    B = len(H[0])
    # Number of subcarriers actually present in H (wideband sweeps pass K > 1
    # while the global config K stays 1, so derive it from the tensor).
    K_eff = H.shape[0]

    # ================= Constant modulus =================
    # NOTE: when F already carries a real-valued selection mask S (sub-connected
    # structure F_eff = F*S with |F| = 1), the division F/|F| collapses the mask
    # because |F*S| = S, so F*S/(S+eps) ~ F — the mask's amplitude and its
    # gradient are erased. skip_unit_modulus keeps the masked magnitudes so the
    # physics loss stays sensitive to S (required for SelectionNet training).
    if not skip_unit_modulus:
        F = F / (torch.abs(F) + 1e-12)

    # ================= Power computation =================
    power = torch.linalg.matrix_norm(F @ W, dim=(-2, -1)) ** 2  # (K, B)

    # Total power per batch
    sum_norm_BB = torch.sum(power, dim=0)  # (B,)
    sum_norm_BB = torch.clamp(sum_norm_BB, min=1e-6)

    # ================= Handle Pt =================
    if torch.is_tensor(Pt) and Pt.dim() >= 1:
        Pt_vec = Pt.to(device=F.device, dtype=sum_norm_BB.dtype)
    else:
        Pt_vec = torch.full((B,), float(Pt), device=F.device, dtype=sum_norm_BB.dtype)

    # ================= Normalize W =================
    normalize_factor = torch.sqrt(K_eff * Pt_vec / sum_norm_BB).view(B, 1, 1)
    W = normalize_factor * W

    return F, W


# ========================= normalize F based on power constraint =====================
def normalize_power(F, W, H, Pt):
    """Normalize F to meet the per-sample power constraint.

    Pt can be a scalar or a 1-D tensor of shape (B,).
    """
    B = len(H[0])
    # Number of subcarriers actually present in H (wideband sweeps pass K > 1
    # while the global config K stays 1, so derive it from the tensor).
    K_eff = H.shape[0]
    sum_norm_power = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)  # (B,)
    sum_norm_power = torch.clamp(sum_norm_power, min=1e-6)
    if torch.is_tensor(Pt) and Pt.dim() >= 1:
        Pt_vec = Pt.to(dtype=sum_norm_power.real.dtype if sum_norm_power.is_complex() else sum_norm_power.dtype,
                       device=F.device)
    else:
        Pt_vec = torch.full((B,), float(Pt), dtype=sum_norm_power.real.dtype if sum_norm_power.is_complex() else sum_norm_power.dtype,
                            device=F.device)
    normalize_factor = torch.sqrt(K_eff * Pt_vec / sum_norm_power).view(B, 1, 1)
    F = normalize_factor * F
    return F

# ========================= re-derive digital precoder for a masked analog F =====================
def compute_digital_precoder(H, F_eff, ridge=1e-2):
    """Differentiable ridge-ZF digital precoder W for the effective channel H F_eff.

    In sub-connected hybrid beamforming the analog F is frozen but only a subset of
    its entries are active (F_eff = F*S). A digital W that was designed for the
    FULL-connected array is structurally mismatched to the masked F_eff, and after
    power normalization the objective becomes (nearly) independent of the mask S —
    which leaves SelectionNet with no learning signal. Re-deriving W for the masked
    effective channel makes the achievable rate/CRB genuinely depend on S.

    H     : (K, B, M, Nt) complex channel
    F_eff : (K, B, Nt, Nrf) masked analog precoder (requires grad through S)
    ridge : relative Tikhonov regularization of the M x M Gram matrix

    Returns W : (K, B, Nrf, M) satisfying H_eff @ W ~ I (ZF property).
    """
    H_eff = torch.einsum('kbmn,kbnj->kbmj', H, F_eff)            # (K, B, M, Nrf)
    G = H_eff @ H_eff.conj().transpose(-1, -2)                  # (K, B, M, M)
    lam = ridge * torch.diagonal(G, dim1=-2, dim2=-1).real.mean().detach()
    I_m = torch.eye(M, dtype=G.dtype, device=G.device)
    W = H_eff.conj().transpose(-1, -2) @ torch.linalg.inv(G + lam * I_m)
    return W


# ========================= generate PC mask =====================
def generage_partial_connection_mask(N, M, device=None):
    mask = torch.zeros((N, M), dtype=COMPLEX_DTYPE, device=device)
    antennas_per_rf = N // M
    for rf in range(M):
        start_idx = rf * antennas_per_rf
        end_idx = start_idx + antennas_per_rf
        mask[start_idx:end_idx, rf] = 1.0 + 0j  # connect these antennas to this RF chain
    return mask


# ========================= generate arbitrary/overlapping PC mask =====================
def build_partial_connection_mask(Nt, Nrf, connections, device=None):
    """Build a (Nt, Nrf) binary connection mask where each RF chain may connect to an
    arbitrary, possibly overlapping and/or wrap-around, contiguous group of antennas.

    Parameters
    ----------
    Nt : int
        Number of transmit antennas.
    Nrf : int
        Number of RF chains.
    connections : sequence of length Nrf
        connections[r] describes which antennas (1-indexed, inclusive) RF chain r
        drives. Each element can be:
          - a single ``(start, end)`` tuple, e.g. ``(1, 20)`` -> antennas 1..20.
            If ``end < start`` the range wraps around the array boundary,
            e.g. ``(48, 12)`` -> antennas 48..64 followed by 1..12.
          - a list of such tuples for chains with multiple disjoint segments,
            e.g. ``[(48, 64), (1, 12)]``.
    device : torch.device, optional

    Returns
    -------
    mask : torch.FloatTensor of shape (Nt, Nrf)

    Example
    -------
    >>> connections = [
    ...     (1, 20),
    ...     (16, 37),
    ...     (32, 53),
    ...     [(48, 64), (1, 12)],
    ... ]
    >>> mask = build_partial_connection_mask(Nt=64, Nrf=4, connections=connections)
    """
    assert len(connections) == Nrf, \
        f"'connections' must have exactly Nrf={Nrf} entries, got {len(connections)}."

    mask = torch.zeros(Nt, Nrf, device=device)

    def _mark_range(rf_idx, start, end):
        start0 = start - 1  # convert to 0-indexed
        end0 = end - 1
        if end0 >= start0:
            idx = torch.arange(start0, end0 + 1)
        else:
            # wrap-around: e.g. start=48, end=12 (1-indexed) on a 64-antenna array
            idx = torch.arange(start0, end0 + 1 + Nt)
        idx = idx % Nt
        mask[idx, rf_idx] = 1.0

    for rf_idx, spec in enumerate(connections):
        segments = [spec] if isinstance(spec, tuple) else spec
        for start, end in segments:
            _mark_range(rf_idx, start, end)

    return mask


# ======================== generate channels =============================================================
def array_response(N, phi, theta):
    # Generate array response vectors
    a = np.zeros([N, 1], dtype='complex_')
    for n in range(N):
        a[n] = (1 / np.sqrt(N)) * np.exp(1j * np.pi * (n * np.sin(phi)))
    return a


def gen_channel(train_batch_size):
    batch_H = np.zeros([K, train_batch_size, M, Nt],
                       dtype='complex64')  # use to save testing data, used latter in Matlab

    for k in range(K):
        for ii in range(train_batch_size):

            # randomly generate azimuth and elevation angles
            AoD = np.zeros([2, Ncluster * Nray], dtype='complex64')
            AoA = np.zeros([2, Ncluster * Nray], dtype='complex64')

            for cc in range(Ncluster):
                AoD_m = np.random.uniform(0, 2 * np.pi, 2)
                AoA_m = np.random.uniform(0, 2 * np.pi, 2)

                AoD[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoD_m[0], angle_sigma, Nray)
                AoD[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoD_m[1], angle_sigma, Nray)
                AoA[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoA_m[0], angle_sigma, Nray)
                AoA[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoA_m[1], angle_sigma, Nray)

            alpha = np.sqrt(sigma / 2) * (
                    np.random.normal(0, 1, Ncluster * Nray) + 1j * np.random.normal(0, 1, Ncluster * Nray))

            # generate channel matrix
            H = np.zeros([M, Nt], dtype='complex_')
            At = np.zeros([Nt, Ncluster * Nray], dtype='complex64')

            for j in range(Ncluster * Nray):
                at = array_response(Nt, AoD[0, j], AoD[1, j])  # UPA array response
                ar = array_response(M, AoA[0, j], AoA[1, j])  # UPA array response
                H = H + alpha[j] * ar * at.conj().T
            H = gamma * H
            batch_H[k, ii, :, :] = H

    return batch_H


# =================================== save generated data ==================================================
def save_data(data_train, data_test):
    # write data
    with h5py.File(data_path_train, 'w') as hf:
        hf.create_dataset('train_set', data=data_train)
    with h5py.File(data_path_test, 'w') as hf:
        hf.create_dataset('test_set', data=data_test)
    # scipy.io.savemat('./channel.mat', {'channel':data_test})


# =================================== load data generated in Matlab ==================================================
def load_data_matlab():
    data_train = scipy.io.loadmat(data_path_train)
    data_train_array = data_train['H_train']
    # data_train_array = data_train['H']
    data_test = scipy.io.loadmat(data_path_test)
    data_test_array = data_test['H_test']
    return data_train_array, data_test_array


# =================================== load data generated in python ==================================================
def load_data():
    # read data
    with h5py.File(data_path_train, 'r') as hf:
        data_train = list(hf.keys())[0]
        data_train_array = hf[data_train][()]
    with h5py.File(data_path_test, 'r') as hf:
        data_test = list(hf.keys())[0]
        data_test_array = hf[data_test][()]
    return data_train_array, data_test_array


# =================================== load data and convert to tensor for trainign=================================
def get_data_tensor(data_source):
    # first load the saved data
    if data_source == 'python':
        data_train_array, data_test_array = load_data()
    else:  # use matlab data
        data_train_array, data_test_array = load_data_matlab()
    # then convert numpy to tensor
    max_train = min(train_size, data_train_array.shape[1])
    max_test = min(test_size, data_test_array.shape[1])

    train_slice = np.ascontiguousarray(data_train_array[:, :max_train, :, :])
    test_slice = np.ascontiguousarray(data_test_array[:, :max_test, :, :])

    H_train_tensor = torch.from_numpy(train_slice).to(COMPLEX_DTYPE).contiguous().to(device)
    H_test_tensor = torch.from_numpy(test_slice).to(COMPLEX_DTYPE).contiguous().to(device)
    return H_train_tensor, H_test_tensor


# /////////////////////////////////////////////////////////////////////////////////////////
#                     WIDEBAND (MULTI-SUBCARRIER) HELPERS
# /////////////////////////////////////////////////////////////////////////////////////////
#
# The wideband OFDM extension treats the existing narrowband channel as the
# reference (center) subcarrier and synthesizes the remaining subcarriers with
# a frequency-selective tapped-delay-line model:
#
#   h_m[k] = sqrt(gamma) * sum_{l=0}^{L-1} sqrt(beta_l) * alpha_{m,l}
#            * a(Nt, theta_{m,l}) * exp(-j 2 pi k tau_l / K)
#
# where beta_l is an exponential power-delay profile and tau_l the normalized
# delay of tap l.  The center subcarrier (k = K//2) is *replaced* by the loaded
# narrowband channel so that K=1 reduces exactly to the legacy behaviour and
# the wideband sweeps stay anchored to the same channel statistics.

def synthesize_wideband_channels(H_ref, n_subcarriers, n_taps=4, delay_spread=3.0, seed=None):
    """Synthesize frequency-selective wideband channels from a narrowband batch.

    Parameters
    ----------
    H_ref : torch.Tensor (1, B, M, Nt) complex
        Narrowband (reference) channel batch, e.g. from ``get_data_tensor``.
    n_subcarriers : int
        Number of OFDM subcarriers K to synthesize.
    n_taps : int
        Number of delay taps in the exponential power-delay profile.
    delay_spread : float
        Controls the normalized tap delays tau_l = l * delay_spread / n_taps.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    H_wb : torch.Tensor (K, B, M, Nt) complex
        Wideband channel tensor; subcarrier ``K//2`` equals ``H_ref``.
    """
    if seed is not None:
        torch.manual_seed(seed)

    K_wb = int(n_subcarriers)
    B = H_ref.shape[1]
    dev = H_ref.device

    # ---- Exponential power-delay profile (normalized to unit total power) ----
    tap_powers = torch.exp(-torch.arange(n_taps, dtype=REAL_DTYPE, device=dev) / (n_taps / 2.0))
    tap_powers = tap_powers / tap_powers.sum()
    tap_delays = torch.arange(n_taps, dtype=REAL_DTYPE, device=dev) * delay_spread / n_taps

    # ---- Per-tap structure: preserve the mmWave sparse spatial geometry ----
    # Each tap shares the reference channel's angle-of-departure structure but
    # gets an independent complex gain (small-scale variation across delay),
    # so the spatial sparsity of the reference channel is retained on every
    # subcarrier.  The dominant tap (l = 0) is exactly the reference channel.
    H_wb = torch.zeros(K_wb, B, H_ref.shape[2], H_ref.shape[3],
                       dtype=COMPLEX_DTYPE, device=dev)

    # Reference channel as the dominant tap response at the center subcarrier.
    H_center = H_ref[0]                                   # (B, M, Nt)

    # Independent complex gains per tap (unit variance), shared across (B, M, Nt).
    tap_gains = [torch.ones((), dtype=COMPLEX_DTYPE, device=dev)]
    for _ in range(n_taps - 1):
        g = randn_complex((1,), device=dev) / np.sqrt(2)
        tap_gains.append(g)

    # Phase rotation across subcarriers: exp(-j 2 pi k tau_l / K)
    k_idx = torch.arange(K_wb, dtype=REAL_DTYPE, device=dev)
    center_k = K_wb // 2

    for k in range(K_wb):
        acc = torch.zeros_like(H_center)
        for l in range(n_taps):
            phase_val = -2.0 * np.pi * float(k - center_k) * float(tap_delays[l]) / max(K_wb, 1)
            phase = torch.complex(torch.cos(torch.tensor(phase_val)),
                                  torch.sin(torch.tensor(phase_val))).to(dev)
            acc = acc + torch.sqrt(tap_powers[l]) * phase * tap_gains[l] * H_center
        H_wb[k] = acc

    # Overwrite the center subcarrier with the exact reference channel so the
    # K=1 case is bit-identical to the legacy narrowband pipeline.
    H_wb[center_k] = H_center

    return H_wb


def build_sensing_matrices_per_subcarrier(n_subcarriers, xi_0_val=1.0):
    """Build per-subcarrier sensing matrices M[k] = A_dot^H(f_k) R_z^-1 A_dot(f_k).

    The steering vector and its angle-derivative are evaluated at each
    subcarrier frequency f_k (normalized around the center carrier):

        a(f_k)   = exp(j 2 pi d n sin(theta) f_k)
        a_dot(f_k) = j 2 pi d n cos(theta) sin(theta) f_k * a(f_k)  (chain rule)

    For a ULA with half-wavelength spacing the frequency scaling enters through
    d = lambda_k / 2 = c / (2 f_k), so the *electrical* angle
    pi * n * sin(theta) * f_k / f_c scales linearly with f_k.

    Parameters
    ----------
    n_subcarriers : int
        Number of subcarriers K.
    xi_0_val : float
        Reference path-loss amplitude (kept for API symmetry; the CRLB metric
        adds log(2 xi_0^2) separately).

    Returns
    -------
    M_k : torch.Tensor (K, Nt, Nt) complex
        Stack of per-subcarrier sensing Fisher matrices.
    A_dot_k : torch.Tensor (K, Nt, Nt) complex
        Stack of per-subcarrier A_dot = a_dot a^H + a a_dot^H matrices.
    """
    K_wb = int(n_subcarriers)
    dev = A_dot.device

    # Normalized subcarrier frequencies around the center carrier: f_k / f_c.
    center_k = K_wb // 2
    freq_scale = torch.ones(K_wb, dtype=REAL_DTYPE, device=dev)
    if K_wb > 1:
        # Fixed total fractional bandwidth (10%) regardless of K, so the
        # per-subcarrier spacing shrinks as K grows — matching a real OFDM
        # system where the occupied bandwidth is fixed and more subcarriers
        # means finer frequency resolution.
        total_bw = 0.10
        freq_scale = 1.0 + total_bw * (torch.arange(K_wb, dtype=REAL_DTYPE, device=dev) - center_k) / K_wb

    n_idx = torch.arange(Nt, dtype=REAL_DTYPE, device=dev)
    sin_t = float(np.sin(desired_angle_rad))
    cos_t = float(np.cos(desired_angle_rad))

    M_list = []
    Adot_list = []
    for k in range(K_wb):
        fs = float(freq_scale[k])
        # Match the system_config convention exactly: a(f) = exp(j 2 pi d n sin(theta))
        # with d = lambda/2, and a_dot(f) = j 2 pi d n cos(theta) * a(f)
        # (derivative w.r.t. theta, NOT including an extra sin factor).
        phase = 1j * 2 * torch.pi * 0.5 * sin_t * fs * n_idx          # d = lambda/2
        a_k = torch.exp(phase).to(COMPLEX_DTYPE)                      # (Nt,)
        adot_k = (1j * 2 * torch.pi * 0.5 * cos_t * fs * n_idx).to(COMPLEX_DTYPE) * a_k

        a_k = a_k.unsqueeze(1)                                        # (Nt, 1)
        adot_k = adot_k.unsqueeze(1)                                  # (Nt, 1)
        A_dot_k = (adot_k @ a_k.transpose(0, 1) + a_k @ adot_k.transpose(0, 1)).to(COMPLEX_DTYPE)
        M_k = (A_dot_k.conj().T @ R_N_inv @ A_dot_k).to(COMPLEX_DTYPE)

        M_list.append(M_k.to(dev))
        Adot_list.append(A_dot_k.to(dev))

    return torch.stack(M_list, dim=0), torch.stack(Adot_list, dim=0)


def get_sum_rate_wideband(H, F, W, Pt):
    """Wideband sum rate: mean over subcarriers of the per-subcarrier rate.

    Implements R = (1/K) sum_k R_k(F, W[k]) with the same per-subcarrier
    interference structure as ``get_sum_rate``.  H, F, W all carry a leading
    subcarrier dimension of size K.

    Returns a scalar tensor (mean over batch and subcarriers).
    """
    return get_sum_rate(H, F, W, Pt)          # get_sum_rate already averages over K


def initialize_wideband(H, Pt):
    """Wideband-aware (F0, W0) initialization for K > 1 subcarriers.

    F0 is frequency-flat: the unit-modulus right-singular vectors of the
    *center* subcarrier channel (matching the legacy 'svd' init).  W0 is a
    per-subcarrier ridge-ZF digital precoder matched to H[k] F0, power
    normalized so that sum_k ||F0 W0[k]||_F^2 = K * Pt (same convention as
    ``normalize``).

    H : (K, B, M, Nt) complex wideband channels.
    Returns (F0, W0) with F0 (K, B, Nt, Nrf) (replicated over K) and
    W0 (K, B, Nrf, M).
    """
    K_wb = H.shape[0]
    B = H.shape[1]
    center_k = K_wb // 2

    # ---- Frequency-flat F0 from the center subcarrier ----
    H_c = H[center_k]                                     # (B, M, Nt)
    # SVD of H^T (B, Nt, M): U columns are the antenna-space right-singular
    # vectors of H, i.e. the dominant beam directions (matching the legacy
    # 'svd' init which takes V[:, :, :, :Nrf] of H directly).
    U_c, _, _ = torch.linalg.svd(H_c.transpose(-2, -1))   # (B, Nt, Nt)
    F0 = U_c[:, :, :Nrf]                                  # (B, Nt, Nrf)
    F0 = F0 / (torch.abs(F0) + 1e-12)
    F0 = F0.unsqueeze(0).expand(K_wb, -1, -1, -1).contiguous()   # (K, B, Nt, Nrf)

    # ---- Per-subcarrier ridge-ZF W0 ----
    H_eff = torch.matmul(H, F0)                           # (K, B, M, Nrf)
    G = H_eff @ H_eff.conj().transpose(-1, -2)            # (K, B, M, M)
    lam = 1e-2 * torch.diagonal(G, dim1=-2, dim2=-1).real.mean().detach()
    I_m = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    W0 = H_eff.conj().transpose(-1, -2) @ torch.linalg.inv(G + lam * I_m)   # (K, B, Nrf, M)

    # ---- Power normalization: per-subcarrier budget ||F0 W0[k]||_F^2 = Pt ----
    # (paper Eq. 10c: |F W[k]|_F^2 = P_BS for every k, so the total across K
    # subcarriers is K * Pt — matching the convention used by ``normalize``.)
    power = torch.linalg.matrix_norm(F0 @ W0, dim=(-2, -1)) ** 2   # (K, B)
    mean_power = power.mean(dim=0).clamp_min(1e-6)                 # (B,)
    if torch.is_tensor(Pt) and Pt.dim() >= 1:
        Pt_vec = Pt.to(device=F0.device, dtype=mean_power.dtype)
    else:
        Pt_vec = torch.full((B,), float(Pt), device=F0.device, dtype=mean_power.dtype)
    scale = torch.sqrt(Pt_vec / mean_power).view(1, B, 1, 1)
    W0 = scale * W0

    return F0, W0


def get_crb_wideband(H, F, W, xi_0_val, M_k, R_N_inv_t, Pt):
    """Wideband log-inverse-CRLB: (1/K) sum_k log(CRLB_k^-1).

    M_k : (K, Nt, Nt) per-subcarrier sensing matrices.
    Returns a (B,) tensor — one entry per sample.
    """
    F, W = normalize(F, W, H, Pt)

    F_H = F.conj().transpose(-2, -1)                       # (K, B, Nrf, Nt)
    W_H = W.conj().transpose(-2, -1)                       # (K, B, M, Nrf)

    inner_mat = W_H @ F_H @ M_k.unsqueeze(1) @ F @ W       # (K, B, M, M)
    batch_trace = torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1)   # (K, B)

    # Per-subcarrier log(CRLB_k^-1), then average over K (Eq. 1 of the paper):
    #   (1/K) sum_k log(2 xi_0^2 * FIM_k)
    fim = batch_trace.real                                  # (K, B)
    crb = torch.log(fim + 1e-12) + torch.log(2 * torch.tensor(xi_0_val, device=fim.device) ** 2)
    return crb.mean(dim=0)                                  # (B,) — average over K


def get_grad_F_crb_wideband(F, W, M_k):
    """Pooled wideband gradient of log(CRLB^-1) w.r.t. F (averaged over K).

    M_k : (K, Nt, Nt).  Returns (K, B, Nt, Nrf) — the per-subcarrier gradient
    stacked over K; the caller weights it with WEIGHT_F_CRB / K when pooling.
    """
    W_H = W.conj().transpose(-2, -1)
    F_H = F.conj().transpose(-2, -1)

    inner_mat = W_H @ F_H @ M_k.unsqueeze(1) @ F @ W       # (K, B, M, M)
    batch_trace = torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1)   # (K, B)
    denom = batch_trace.view(F.shape[0], -1, 1, 1)

    numerator = M_k.unsqueeze(1) @ F @ W @ W_H             # (K, B, Nt, Nrf)
    return numerator / (denom + 1e-12)


def get_grad_W_crb_wideband(F, W, M_k):
    """Per-subcarrier gradient of log(CRLB^-1) w.r.t. W[k].

    M_k : (K, Nt, Nt).  Returns (K, B, Nrf, M).
    """
    W_H = W.conj().transpose(-2, -1)
    F_H = F.conj().transpose(-2, -1)

    inner_mat = W_H @ F_H @ M_k.unsqueeze(1) @ F @ W       # (K, B, M, M)
    batch_trace = torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1)   # (K, B)
    denom = batch_trace.view(F.shape[0], -1, 1, 1)

    numerator = F_H @ M_k.unsqueeze(1) @ F @ W             # (K, B, Nrf, M)
    return numerator / (denom + 1e-12)


# =================================== load radar data generated in Matlab ==================================================
def get_radar_data(snr_dB, H):
    """Load pre-computed radar covariance matrix R and array-steering vectors.

    snr_dB can be:
      - a scalar (float / int / 0-d array) — same R replicated over the batch (original behaviour)
      - a 1-D array of shape (B,)          — per-sample R assembled from the lookup table

    Returns
    -------
    R          : torch tensor  (K, B, Nt, Nt)
    at         : torch tensor  (K, B, Nt, n_angles)
    theta      : np.ndarray    angle grid
    ideal_beam : np.ndarray    ideal beampattern
    """
    radar_data_file_name = directory_data + 'radar_data.mat'
    radar_data = scipy.io.loadmat(radar_data_file_name)
    R0_4D = radar_data['J']          # shape: (Nt, Nt, K_freq, n_snr)  or (Nt, Nt, n_snr) when K==1

    at_2D = radar_data['a']
    theta = radar_data['theta']
    ideal_beam = radar_data['Pd_theta']

    B = len(H[0])

    # ------------------------------------------------------------------ #
    # Determine whether snr_dB is a scalar or per-sample array
    # ------------------------------------------------------------------ #
    snr_dB_arr = np.atleast_1d(np.asarray(snr_dB, dtype=float)).ravel()
    per_sample = snr_dB_arr.size > 1   # True  → per-sample path
                                        # False → replicate-scalar path

    if K == 1:
        if per_sample:
            # Build R per sample: shape (1, B, Nt, Nt)
            R_list = []
            for s in snr_dB_arr:
                idx = np.where(snr_dB_list == s)[0]
                R_s = np.squeeze(R0_4D[:, :, 0, idx])   # (Nt, Nt)
                R_list.append(R_s[None, :, :])           # (1, Nt, Nt)
            R_stack = np.stack(R_list, axis=0)           # (B, Nt, Nt)
            R_array = R_stack[None, :, :, :]             # (1, B, Nt, Nt)
        else:
            idx_snr = np.where(snr_dB_list == snr_dB_arr[0])
            R0_2D = np.squeeze(R0_4D[:, :, 0, idx_snr])
            R_array = np.tile(R0_2D, [1, B, 1, 1])

        at0 = np.expand_dims(at_2D, axis=0)
        at_array1 = np.tile(at0, (B, 1, 1, 1))
        at_array = np.transpose(at_array1, (1, 0, 2, 3))
    else:
        if per_sample:
            R_list = []
            for s in snr_dB_arr:
                idx = np.where(snr_dB_list == s)[0]
                R_s = np.squeeze(R0_4D[:, :, :, idx])   # (Nt, Nt, K)
                R_s_k = np.transpose(R_s, (2, 0, 1))    # (K, Nt, Nt)
                R_list.append(R_s_k[:, None, :, :])     # (K, 1, Nt, Nt)
            R_stack = np.concatenate(R_list, axis=1)    # (K, B, Nt, Nt)
            R_array = R_stack
        else:
            idx_snr = np.where(snr_dB_list == snr_dB_arr[0])
            R0_2D = np.squeeze(R0_4D[:, :, :, idx_snr])
            R_array0 = np.transpose(R0_2D, (2, 0, 1))
            R_array1 = np.tile(R_array0, [B, 1, 1, 1])
            R_array = np.transpose(R_array1, (1, 0, 2, 3))

        at0 = np.transpose(at_2D, (2, 0, 1))
        at_array1 = np.tile(at0, (B, 1, 1, 1))
        at_array = np.transpose(at_array1, (1, 0, 2, 3))

    R = torch.from_numpy(R_array).to(COMPLEX_DTYPE).contiguous().to(device)
    at = torch.from_numpy(at_array).to(COMPLEX_DTYPE).contiguous().to(device)

    return R, at, theta, ideal_beam[0, :]


# =================================== compute the power consumption of the hybrid precoding system ==================================================
def get_power_consumption(F, W, mask, P_RF=P_RF, P_PS=P_PS, eta=PA_EFFICIENCY):
    # 1.  Compute the transmit power of the hybrid precoding system
    transmit_power = torch.linalg.matrix_norm(( F * mask ) @ W, ord='fro') ** 2  # (K, B)
    transmit_power = (transmit_power.sum(dim=0).real) / eta            # (B,)

    # ================= Active-connection mask =================
    mask_active = torch.abs(mask) > 0                      # (Nt, Nrf) or (B, Nt, Nrf)
    if mask_active.dim() == 2:
        mask_active = mask_active.unsqueeze(0)              # (1, Nt, Nrf) -> broadcast over batch

    # 2. Compute the RF chain power consumption
    # rf_chain_power = number of non-zero columns in mask x P_rf (power consumption of a single RF chain)
    n_rf_active = mask_active.any(dim=-2).sum(dim=-1).to(transmit_power.dtype)  # (1,) or (B,)
    rf_chain_power = n_rf_active * P_RF

    # 3. Calculate the phase shifters power consumption
    # phase_shifter_power = number of non-zero elements in mask x P_ps (power consumption of a single phase shifter)
    n_ps_active = mask_active.sum(dim=(-2, -1)).to(transmit_power.dtype)        # (1,) or (B,)
    phase_shifter_power = n_ps_active * P_PS

    # 4. Calculate the total power consumption
    total_power_consumption = transmit_power + rf_chain_power + phase_shifter_power

    return total_power_consumption



# =================================== get the array of beampattern values ==================================================
def get_beampattern(F, W, at, Pt):
    Q = F @ W
    at_H = torch.transpose(at, 2, 3).conj()
    Q_H = torch.transpose(Q, 2, 3).conj()
    B = at_H @ Q @ Q_H @ at
    # print(torch.linalg.matrix_norm(B, ord='fro') ** 2)
    Bdiag = torch.diagonal(B, offset=0, dim1=-1, dim2=-2) / Pt
    # Bmean = 10 * torch.log10(torch.real(torch.mean(Bdiag, 1)))
    Bmean = torch.real(torch.mean(torch.mean(Bdiag, 1), 0))
    B_array = Bmean.detach().cpu().numpy()
    return B_array

# if __name__ == '__main__':
#     # generate data
#     channel_train = gen_channel(train_size)
#     channel_test = gen_channel(test_size)
#
#     # save data
#     save_data(channel_train, channel_valid, channel_test)
#     data_train_array, data_test_array = load_data()
#
#     get_data_tensor()

# print(channel_train[0][0])
# print(data_train_array[0][0])
# print('------------------------------')
# print(channel_test[0][0])
# print(data_test_array[0][0])
# print('------------------------------')

def extract_active_elements(F):
    """
    F: (B, 1, Nt, Nrf) complex
    Returns:
        F_active: (B, Nt, 1) complex
    """
    B, _, Nt, Nrf = F.shape
    aprf = Nt // Nrf
    F_2d = F[:, 0, :, :]                               # (B, Nt, Nrf)
    F_blocks = F_2d.reshape(B, Nrf, aprf, Nrf)          # (B, Nrf, aprf, Nrf)
    F_diag = torch.diagonal(F_blocks, dim1=1, dim2=3)   # (B, aprf, Nrf)
    F_active = F_diag.permute(0, 2, 1).reshape(B, Nt, 1)
    return F_active
    
def safe_legend(**kwargs):
    """Add legend only when labeled artists exist to avoid Matplotlib warnings."""
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    valid = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith('_')]
    if not valid:
        return
    valid_handles, valid_labels = zip(*valid)
    # Use opaque legend frame to keep EPS exports warning-free.
    kwargs.setdefault('framealpha', 1.0)
    ax.legend(valid_handles, valid_labels, **kwargs)