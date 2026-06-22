from algorithms import *
import scipy.io
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from utility import safe_legend

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

benchmark = 1
# torch.manual_seed(3407)

# ///////////////////////////////////////// SHOW RATES VS. SNRs ///////////////////////////////////
snr_dB_list = np.array([0,2,4,6,8,10,12], dtype='float64')
# Load training data
_, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]



# Create new model for conentional PGA
if run_conv_PGA == 1:
    model_conv_PGA = PGA_Unfold_JX(step_size_UPGA_J1)
if run_conv_PGA_J5 == 1:
    conv_PGA_J5 = PGA_Unfold_JX(torch.full_like(step_size_UPGA_J5, step_size_fixed, requires_grad=False,))
if run_conv_PGA_J10 == 1:
    conv_PGA_J10 = PGA_Unfold_JX(torch.full_like(step_size_UPGA_J10, step_size_fixed, requires_grad=False,))


# Create new model and load states for unfolded PGA with different J values
if run_UPGA_J1 == 1:
    model_UPGA_J1 = PGA_Unfold_JX(step_size_UPGA_J1)
    model_UPGA_J1.load_state_dict(torch.load(model_file_name_UPGA_J1, map_location=device))
if run_UPGA_J4 == 1:
    model_UPGA_J4 = PGA_Unfold_JX(step_size_UPGA_J4)
    model_UPGA_J4.load_state_dict(torch.load(directory_model + 'UPGA_J4.pth', map_location=device))
if run_UPGA_J5 == 1:
    model_UPGA_J5 = PGA_Unfold_JX(step_size_UPGA_J5)
    model_UPGA_J5.load_state_dict(torch.load(model_file_name_UPGA_J5, map_location=device))
if run_UPGA_J6 == 1:
    model_UPGA_J6 = PGA_Unfold_JX(step_size_UPGA_J6)
    model_UPGA_J6.load_state_dict(torch.load(directory_model + 'UPGA_J6.pth', map_location=device))
if run_UPGA_J10 == 1:
    model_UPGA_J10 = PGA_Unfold_JX(step_size_UPGA_J10)
    model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10, map_location=device))
if run_UPGA_J20 == 1:
    model_UPGA_J20 = PGA_Unfold_JX(step_size_UPGA_J20)
    model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))


# Create new model and load states for unfolded PGA with decay
if run_UPGA_J5_decay == 1:
    model_UPGA_J5_decay = PGA_Unfold_JX_decay(step_size_UPGA_J5_decay)
    model_UPGA_J5_decay.load_state_dict(torch.load(model_file_name_UPGA_J5_decay, map_location=device))
if run_UPGA_J10_decay == 1:
    model_UPGA_J10_decay = PGA_Unfold_JX_decay(step_size_UPGA_J10_decay)
    model_UPGA_J10_decay.load_state_dict(torch.load(model_file_name_UPGA_J10_decay, map_location=device))
if run_UPGA_J20_decay == 1:
    model_UPGA_J20_decay = PGA_Unfold_JX_decay(step_size_UPGA_J20_decay)
    model_UPGA_J20_decay.load_state_dict(torch.load(model_file_name_UPGA_J20_decay, map_location=device))


# if run_UPGA_J_GradReuse == 1:
#     model_UPGA_J_GradReuse = PGA_Unfold_J_GradReuse(step_size_UPGA_J_GradReuse)
#     model_UPGA_J_GradReuse.load_state_dict(torch.load(model_file_name_UPGA_J_GradReuse, map_location=device))


rate_conv_PGA = np.zeros([len(snr_dB_list), ], dtype=float)
rate_conv_PGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_conv_PGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)

rate_UPGA_J4 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J6 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype=float)

rate_UPGA_J5_decay = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J10_decay = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J20_decay = np.zeros([len(snr_dB_list), ], dtype=float)

# rate_UPGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)
# rate_conv_PGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)
# rate_UPGA_J_GradReuse = np.zeros([len(snr_dB_list), ], dtype=float)

# CRB-based sensing metrics for new models (PGA_Unfold_J10, PGA_Unfold_J20)
CRB_conv_PGA = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_conv_PGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_conv_PGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)

CRB_UPGA_J4 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J6 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype=float)

CRB_UPGA_J5_decay = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J10_decay = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J20_decay = np.zeros([len(snr_dB_list), ], dtype=float)

# CRB_UPGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)
# CRB_conv_PGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)

# CRB_UPGA_J_GradReuse = np.zeros([len(snr_dB_list), ], dtype=float)

for ss in range(len(snr_dB_list)):
    snr_dB = snr_dB_list[ss]
    snr_ss = 10 ** (snr_dB / 10)
    print('---------------------- snr = ' + str(snr_dB))

    # load data
    R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)

    # Conventional PGA ====================================
    if run_conv_PGA == 1:
        rate_conv_PGA[ss], CRB_conv_PGA[ss] = execute_conv_PGA(model_conv_PGA, H_test, snr_ss)
    if run_conv_PGA_J5 == 1:
        rate_conv_PGA_J5[ss], CRB_conv_PGA_J5[ss] = execute_conv_PGA_J5(conv_PGA_J5, H_test, snr_ss)
    if run_conv_PGA_J10 == 1:
        rate_conv_PGA_J10[ss], CRB_conv_PGA_J10[ss] = execute_conv_PGA_J10(conv_PGA_J10, H_test, snr_ss)

    if run_UPGA_J4 == 1:
        rate_UPGA_J4[ss], CRB_UPGA_J4[ss] = execute_UPGA_J4(model_UPGA_J4, H_test, snr_ss)
    if run_UPGA_J5 == 1:
        rate_UPGA_J5[ss], CRB_UPGA_J5[ss] = execute_UPGA_J5(model_UPGA_J5, H_test, snr_ss)
    if run_UPGA_J6 == 1:
        rate_UPGA_J6[ss], CRB_UPGA_J6[ss] = execute_UPGA_J6(model_UPGA_J6, H_test, snr_ss)
    if run_UPGA_J10 == 1:
        rate_UPGA_J10[ss], CRB_UPGA_J10[ss] = execute_UPGA_J10(model_UPGA_J10, H_test, snr_ss)
    if run_UPGA_J20 == 1:
        rate_UPGA_J20[ss], CRB_UPGA_J20[ss] = execute_UPGA_J20(model_UPGA_J20, H_test, snr_ss)

    if run_UPGA_J5_decay == 1:
        rate_UPGA_J5_decay[ss], CRB_UPGA_J5_decay[ss] = execute_UPGA_J5_decay(model_UPGA_J5_decay, H_test, snr_ss)
    if run_UPGA_J10_decay == 1:
        rate_UPGA_J10_decay[ss], CRB_UPGA_J10_decay[ss] = execute_UPGA_J10_decay(model_UPGA_J10_decay, H_test, snr_ss)
    if run_UPGA_J20_decay == 1:
        rate_UPGA_J20_decay[ss], CRB_UPGA_J20_decay[ss] = execute_UPGA_J20_decay(model_UPGA_J20_decay, H_test, snr_ss)
    # if run_UPGA_J_GradReuse == 1:
    #     rate_UPGA_J_GradReuse[ss], CRB_UPGA_J_GradReuse[ss] = execute_UPGA_J_GradReuse(model_UPGA_J_GradReuse, H_test, snr_ss)


# ==========================plot rate vs SNR ======================================================
plt.figure(figsize=(8, 4.2))
# plt.rcParams["figure.figsize"] = (6.5, 3.2)
if run_conv_PGA == 1:
    plt.plot(snr_dB_list, rate_conv_PGA, '--', color='black', linewidth=3, markersize=8, label=Conv_PGA_J1)
if run_conv_PGA_J5 == 1:
    plt.plot(snr_dB_list, rate_conv_PGA_J5, '--', color='blue', linewidth=3, markersize=8, label=Conv_PGA_J5)
if run_conv_PGA_J10 == 1:
    plt.plot(snr_dB_list, rate_conv_PGA_J10, '-*', color='blue', linewidth=3, markersize=8, label=Conv_PGA_J10)
if run_UPGA_J4 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J4, '--', color='orange', linewidth=3, markersize=8, label=label_UPGA_J4)
if run_UPGA_J5 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J5, '--', color='red', linewidth=3, markersize=8, label=label_UPGA_J5)
if run_UPGA_J6 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J6, '-d', color='orange', linewidth=3, markersize=8, label=label_UPGA_J6)
if run_UPGA_J10 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J10, '-*', color='red', linewidth=3, markersize=8, label=label_UPGA_J10)
if run_UPGA_J20 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J20, '-*', color='red', linewidth=3, markersize=8, label=label_UPGA_J20)
    plt.plot(snr_dB_list, rate_conv_PGA, ':', color='black', linewidth=3, markersize=8, label=label_conv)
if run_UPGA_J5_decay == 1:
    plt.plot(snr_dB_list, rate_UPGA_J5_decay, '--', color='green', linewidth=3, markersize=8, label=label_UPGA_J5_decay)
if run_UPGA_J10_decay == 1:
    plt.plot(snr_dB_list, rate_UPGA_J10_decay, '-*', color='green', linewidth=3, markersize=8, label=label_UPGA_J10_decay)
if run_UPGA_J20_decay == 1:
    plt.plot(snr_dB_list, rate_UPGA_J20_decay, ':p', color='green', linewidth=3, markersize=8, label=label_UPGA_J20_decay)

# system_params = '$N=' + str(Nt) + ', M=' + str(M) + ', N_{\\mathrm{RF}}=' + str(Nrf) + ', \\omega=' + str(OMEGA) + '$'
# plt.title(system_params)
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.grid()
# safe_legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=11, labelspacing=0.1, ncol=2, frameon=False, columnspacing=0.6,)
plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png', bbox_inches='tight', pad_inches=0.02)
plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps', bbox_inches='tight', pad_inches=0.02)


#============================== plot CRB vs SNR ======================================================

# ======================= Plot CRLB vs SNR with inset =======================

plt.figure(figsize=(8, 4.2))
ax = plt.gca()

def crlb_from_log_inv(x):
    return (1 / torch.exp(torch.tensor(x))).detach().cpu().numpy()

def plot_curve(x, y, style, color, label=None, lw=3, ms=7, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, style, color=color, linewidth=lw, markersize=ms, label=label)

curves = []

if run_conv_PGA == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_conv_PGA), '--', 'black', Conv_PGA_J1))

if run_conv_PGA_J5 == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_conv_PGA_J5), '--', 'blue', Conv_PGA_J5))

if run_conv_PGA_J10 == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_conv_PGA_J10), '-*', 'blue', Conv_PGA_J10))

if run_UPGA_J4 == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J4), '--', 'orange', label_UPGA_J4))

if run_UPGA_J5 == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J5), '--', 'red', label_UPGA_J5))

if run_UPGA_J6 == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J6), '-d', 'orange', label_UPGA_J6))

if run_UPGA_J10 == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J10), '-*', 'red', label_UPGA_J10))

if run_UPGA_J20 == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J20), ':s', 'red', label_UPGA_J20))

if run_UPGA_J5_decay == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J5_decay), '--', 'green', label_UPGA_J5_decay))

if run_UPGA_J10_decay == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J10_decay), '-*', 'green', label_UPGA_J10_decay))

if run_UPGA_J20_decay == 1:
    curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J20_decay), ':p', 'green', label_UPGA_J20_decay))

# Main plot
for x, y, style, color, label in curves:
    plot_curve(x, y, style, color, label=label, ax=ax)

ax.set_xlabel('SNR [dB]', fontsize=14)
ax.set_ylabel(r'$\mathrm{CRLB}$', fontsize=14)
ax.grid(True)
# ax.legend(loc='upper right', labelspacing=0.15, fontsize=12)

# Inset zoom
axins = inset_axes(ax, width="42%", height="42%", loc='upper right', borderpad=1.2)

for x, y, style, color, label in curves:
    plot_curve(x, y, style, color, ax=axins, lw=2, ms=5)

axins.set_xlim(10, 12)

zoom_vals = []
for x, y, style, color, label in curves:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    mask = (x_arr >= 10) & (x_arr <= 12)
    if np.any(mask):
        zoom_vals.extend(y_arr[mask])

if len(zoom_vals) > 0:
    ymin = np.min(zoom_vals)
    ymax = np.max(zoom_vals)
    ypad = 0.15 * (ymax - ymin) if ymax > ymin else 0.1 * ymax
    axins.set_ylim(ymin - ypad, ymax + ypad)

axins.grid(True)
axins.tick_params(labelsize=8)
axins.patch.set_alpha(1.0)
axins.patch.set_facecolor("white")

mark_inset(ax, axins, loc1=2, loc2=4, fc="white", ec="0.4", linewidth=1)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=12, labelspacing=0.1, ncol=2, frameon=False, columnspacing=0.6)
plt.savefig(directory_result + 'CRB_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png', bbox_inches='tight', pad_inches=0.02)
plt.savefig(directory_result + 'CRB_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps', bbox_inches='tight', pad_inches=0.02)
# # Save SNR-curve data for MATLAB plotting (rate, CRB, and objective)
# print('Saving SNR-curve results to .mat file...')
# mat_data = {'snr_dB_list': snr_dB_list}

# if run_conv_PGA == 1:
#     mat_data['rate_conv_PGA_J1'] = rate_conv_PGA
#     mat_data['crb_conv_PGA_J1'] = CRB_conv_PGA
#     mat_data['obj_conv_PGA_J1'] = OMEGA * rate_conv_PGA + CRB_conv_PGA
# if run_conv_PGA_J5 == 1:
#     mat_data['rate_conv_PGA_J5'] = rate_conv_PGA_J5
#     mat_data['crb_conv_PGA_J5'] = CRB_conv_PGA_J5
#     mat_data['obj_conv_PGA_J5'] = OMEGA * rate_conv_PGA_J5 + CRB_conv_PGA_J5
# if run_conv_PGA_J10 == 1:
#     mat_data['rate_conv_PGA_J10'] = rate_conv_PGA_J10
#     mat_data['crb_conv_PGA_J10'] = CRB_conv_PGA_J10
#     mat_data['obj_conv_PGA_J10'] = OMEGA * rate_conv_PGA_J10 + CRB_conv_PGA_J10


# if run_UPGA_J5 == 1:
#     mat_data['rate_UPGA_J5'] = rate_UPGA_J5
#     mat_data['crb_UPGA_J5'] = CRB_UPGA_J5
#     mat_data['obj_UPGA_J5'] = OMEGA * rate_UPGA_J5 + CRB_UPGA_J5
# if run_UPGA_J10 == 1:
#     mat_data['rate_UPGA_J10'] = rate_UPGA_J10
#     mat_data['crb_UPGA_J10'] = CRB_UPGA_J10
#     mat_data['obj_UPGA_J10'] = OMEGA * rate_UPGA_J10 + CRB_UPGA_J10
# if run_UPGA_J20 == 1:
#     mat_data['rate_UPGA_J20'] = rate_UPGA_J20
#     mat_data['crb_UPGA_J20'] = CRB_UPGA_J20
#     mat_data['obj_UPGA_J20'] = OMEGA * rate_UPGA_J20 + CRB_UPGA_J20

# if run_UPGA_J5_decay == 1:
#     mat_data['rate_UPGA_J5_decay'] = rate_UPGA_J5_decay
#     mat_data['crb_UPGA_J5_decay'] = CRB_UPGA_J5_decay
#     mat_data['obj_UPGA_J5_decay'] = OMEGA * rate_UPGA_J5_decay + CRB_UPGA_J5_decay
# if run_UPGA_J10_decay == 1:
#     mat_data['rate_UPGA_J10_decay'] = rate_UPGA_J10_decay
#     mat_data['crb_UPGA_J10_decay'] = CRB_UPGA_J10_decay
#     mat_data['obj_UPGA_J10_decay'] = OMEGA * rate_UPGA_J10_decay + CRB_UPGA_J10_decay
# if run_UPGA_J20_decay == 1:
#     mat_data['rate_UPGA_J20_decay'] = rate_UPGA_J20_decay
#     mat_data['crb_UPGA_J20_decay'] = CRB_UPGA_J20_decay
#     mat_data['obj_UPGA_J20_decay'] = OMEGA * rate_UPGA_J20_decay + CRB_UPGA_J20_decay
# if run_UPGA_J_GradReuse == 1:
#     mat_data['rate_UPGA_J_GradReuse'] = rate_UPGA_J_GradReuse
#     mat_data['crb_UPGA_J_GradReuse'] = CRB_UPGA_J_GradReuse
#     mat_data['obj_UPGA_J_GradReuse'] = OMEGA * rate_UPGA_J_GradReuse + CRB_UPGA_J_GradReuse

# mat_file_name = directory_result + 'snr_results_' + str(Nt) + '_' + str(OMEGA) + '.mat'
# scipy.io.savemat(mat_file_name, mat_data)
# print(f'  Saved to {mat_file_name}')



# plt.show()
