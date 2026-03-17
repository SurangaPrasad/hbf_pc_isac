
from utility import *
from PGA_models import *

# load at for old MSE-based models (conv_PGA, UPGA_J10_PC, etc.)
_, H_test_tmp = get_data_tensor(data_source)
H_test_tmp1 = H_test_tmp[:, :test_size, :, :]
R_tmp, at0, _, ideal_beam = get_radar_data(snr_dB, H_test_tmp1)
at = at0[:, : test_size, :, :]

def execute_conv_PGA(conv_PGA, H_test, R, Pt):
    # executing classical PGA on the test set
    rate, tau, F, W = conv_PGA.execute_PGA(H_test, R, Pt, n_iter_outer)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr

def execute_UPGA_J1(model_UPJA_J1, H_test, R, Pt):
    rate, tau, F, W = model_UPJA_J1.execute_PGA(H_test, R, Pt, n_iter_outer)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr

def execute_UPGA_J20(model_UPGA_J20, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J20.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J20)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J10(model_UPGA_J10, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J10_PC(model_UPJA_J10_PC, H_test, R, Pt):
    rate, tau, F, W = model_UPJA_J10_PC.execute_PGA(H_test, R, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr

def execute_conv_PGA_J10_PC(conv_PGA_J10_PC, H_test, R, Pt):
    # executing classical PGA on the test set
    rate, tau, F, W = conv_PGA_J10_PC.execute_PGA(H_test, R, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr

def execute_conv_PGA_J10(conv_PGA_J10, H_test, Pt):
    # Conventional PGA with J=10 inner iterations (CRB-based sensing metric)
    rates, crbs, _, F, W = conv_PGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J_decay(model_UPGA_J_decay, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J_decay.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr