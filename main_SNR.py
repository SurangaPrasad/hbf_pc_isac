from algorithms import *
import scipy.io

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

benchmark = 1
# torch.manual_seed(3407)

# ///////////////////////////////////////// SHOW RATES VS. SNRs ///////////////////////////////////
snr_dB_list = np.array([0,2,4,6,8,10,12], dtype='float64')
run_UPGA_J20 = 1 
# Load training data
_, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]
# Create new model and load states
if run_conv_PGA == 1:
    model_conv_PGA = PGA_Unfold_JX(step_size_UPGA_J1)
if run_UPGA_J1 == 1:
    model_UPGA_J1 = PGA_Unfold_JX(step_size_UPGA_J1)
    model_UPGA_J1.load_state_dict(torch.load(model_file_name_UPGA_J1, map_location=device))
if run_UPGA_J20 == 1:
    model_UPGA_J20 = PGA_Unfold_JX(step_size_UPGA_J20)
    model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))
if run_UPGA_J10 == 1:
    model_UPGA_J10 = PGA_Unfold_JX(step_size_UPGA_J10)
    model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10, map_location=device))
if run_conv_PGA_J10 == 1:
    conv_PGA_J10 = PGA_Unfold_JX(torch.full_like(step_size_UPGA_J10, step_size_fixed, requires_grad=False,))
if run_UPGA_J10_decay == 1:
    model_UPGA_J10_decay = PGA_Unfold_JX_decay(step_size_UPGA_J10_decay)
    model_UPGA_J10_decay.load_state_dict(torch.load(model_file_name_UPGA_J10_decay, map_location=device))
if run_UPGA_J20_decay == 1:
    model_UPGA_J20_decay = PGA_Unfold_JX_decay(step_size_UPGA_J20_decay)
    model_UPGA_J20_decay.load_state_dict(torch.load(model_file_name_UPGA_J20_decay, map_location=device))
if run_UPGA_J_GradReuse == 1:
    model_UPGA_J_GradReuse = PGA_Unfold_J_GradReuse(step_size_UPGA_J_GradReuse)
    model_UPGA_J_GradReuse.load_state_dict(torch.load(model_file_name_UPGA_J_GradReuse, map_location=device))


rate_conv_PGA = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J1 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)
rate_conv_PGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)
rate_conv_PGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J10_decay = np.zeros([len(snr_dB_list), ], dtype=float)
rate_UPGA_J20_decay = np.zeros([len(snr_dB_list), ], dtype=float)

rate_UPGA_J_GradReuse = np.zeros([len(snr_dB_list), ], dtype=float)

# CRB-based sensing metrics for new models (PGA_Unfold_J10, PGA_Unfold_J20)
CRB_conv_PGA = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J1 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_conv_PGA_J10_PC = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_conv_PGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J10_decay = np.zeros([len(snr_dB_list), ], dtype=float)
CRB_UPGA_J20_decay = np.zeros([len(snr_dB_list), ], dtype=float)

CRB_UPGA_J_GradReuse = np.zeros([len(snr_dB_list), ], dtype=float)

for ss in range(len(snr_dB_list)):
    snr_dB = snr_dB_list[ss]
    snr_ss = 10 ** (snr_dB / 10)
    print('---------------------- snr = ' + str(snr_dB))

    # load data
    R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)

    # Conventional PGA ====================================
    if run_conv_PGA == 1:
        rate_conv_PGA[ss], CRB_conv_PGA[ss] = execute_conv_PGA(model_conv_PGA, H_test, snr_ss)
    if run_UPGA_J1 == 1:
        rate_UPGA_J1[ss], CRB_UPGA_J1[ss] = execute_UPGA_J1(model_UPGA_J1, H_test, snr_ss)
    if run_UPGA_J10 == 1:
        rate_UPGA_J10[ss], CRB_UPGA_J10[ss] = execute_UPGA_J10(model_UPGA_J10, H_test, snr_ss)
    if run_UPGA_J20 == 1:
        rate_UPGA_J20[ss], CRB_UPGA_J20[ss] = execute_UPGA_J20(model_UPGA_J20, H_test, snr_ss)
    if run_conv_PGA_J10 == 1:
        rate_conv_PGA_J10[ss], CRB_conv_PGA_J10[ss] = execute_conv_PGA_J10(conv_PGA_J10, H_test, snr_ss)
    if run_UPGA_J10_decay == 1:
        rate_UPGA_J10_decay[ss], CRB_UPGA_J10_decay[ss] = execute_UPGA_J10_decay(model_UPGA_J10_decay, H_test, snr_ss)
    if run_UPGA_J20_decay == 1:
        rate_UPGA_J20_decay[ss], CRB_UPGA_J20_decay[ss] = execute_UPGA_J20_decay(model_UPGA_J20_decay, H_test, snr_ss)
    if run_UPGA_J_GradReuse == 1:
        rate_UPGA_J_GradReuse[ss], CRB_UPGA_J_GradReuse[ss] = execute_UPGA_J_GradReuse(model_UPGA_J_GradReuse, H_test, snr_ss)


# plot rate vs SNR ======================================================
fig_rate = plt.figure(1)
plt.rcParams["figure.figsize"] = (6.4, 4.0)
if run_conv_PGA == 1:
    plt.plot(snr_dB_list, rate_conv_PGA, '--', color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
if run_UPGA_J1 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J1, '-o', color='cyan', linewidth=3, markersize=7, label=label_UPGA_J1)
if run_UPGA_J10 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J10, ':*', color='red', linewidth=3, markersize=7, label=label_UPGA_J10)
if run_UPGA_J20 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J20, '-', color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
if run_conv_PGA == 1:
    plt.plot(snr_dB_list, rate_conv_PGA, ':', color='black', linewidth=3, markersize=7, label=label_conv)
if run_UPGA_J10_PC == 1:
    plt.plot(snr_dB_list, rate_UPGA_J10_PC, ':', color='green', linewidth=3, markersize=7, label=label_UPGA_J10_PC)
if run_conv_PGA_J10_PC == 1:
    plt.plot(snr_dB_list, rate_conv_PGA_J10_PC, ':', color='orange', linewidth=3, markersize=7, label=label_conv_PGA_J10_PC)
if run_conv_PGA_J10 == 1:
    plt.plot(snr_dB_list, rate_conv_PGA_J10, '--', color='green', linewidth=3, markersize=7, label=label_PGA_J10)
if run_UPGA_J10_decay == 1:
    plt.plot(snr_dB_list, rate_UPGA_J10_decay, ':d', color='purple', linewidth=3, markersize=7, label=label_UPGA_J10_decay)
if run_UPGA_J20_decay == 1:
    plt.plot(snr_dB_list, rate_UPGA_J20_decay, ':p', color='brown', linewidth=3, markersize=7, label=label_UPGA_J20_decay)
if run_UPGA_J_GradReuse == 1:
    plt.plot(snr_dB_list, rate_UPGA_J_GradReuse, ':^', color='teal', linewidth=3, markersize=7, label=label_UPGA_J_GradReuse)
# if benchmark == 1:
#     plt.plot(snr_dB_list, rate_SCA, '-x', color='black', linewidth=3, markersize=7, label=label_SCA)
#     plt.plot(snr_dB_list, rate_ZF, '-o', color='purple', linewidth=3, markersize=7, label=label_ZF)

system_params = '$N=' + str(Nt) + ', M=' + str(M) + ', N_{\\mathrm{RF}}=' + str(Nrf) + ', \\omega=' + str(OMEGA) + '$'
# plt.title(system_params)
plt.xlabel('SNR [dB]')
plt.ylabel(r'$R$ [bits/s/Hz]')
plt.grid()
plt.legend(loc='upper left', labelspacing  = 0.15)
plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png')
plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps')



plt.show()

# plot CRB vs SNR ======================================================
fig_CRB = plt.figure(2)
plt.rcParams["figure.figsize"] = (6.4, 4.0)
if run_conv_PGA == 1:
    plt.plot(snr_dB_list, CRB_conv_PGA, '--', color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
if run_UPGA_J1 == 1:
    plt.plot(snr_dB_list, CRB_UPGA_J1, '-o', color='cyan', linewidth=3, markersize=7, label=label_UPGA_J1)
if run_UPGA_J10 == 1:
    plt.plot(snr_dB_list, CRB_UPGA_J10, ':*', color='red', linewidth=3, markersize=7, label=label_UPGA_J10)
if run_UPGA_J20 == 1:
    plt.plot(snr_dB_list, CRB_UPGA_J20, '-', color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
if run_conv_PGA == 1:
    plt.plot(snr_dB_list, CRB_conv_PGA, ':', color='black', linewidth=3, markersize=7, label=label_conv)
if run_UPGA_J10_PC == 1:
    plt.plot(snr_dB_list, CRB_UPGA_J10_PC, ':', color='green', linewidth=3, markersize=7, label=label_UPGA_J10_PC)
if run_conv_PGA_J10_PC == 1:
    plt.plot(snr_dB_list, CRB_conv_PGA_J10_PC, ':', color='orange', linewidth=3, markersize=7, label=label_conv_PGA_J10_PC)
if run_conv_PGA_J10 == 1:
    plt.plot(snr_dB_list, CRB_conv_PGA_J10, '--', color='green', linewidth=3, markersize=7, label=label_PGA_J10)
if run_UPGA_J10_decay == 1:
    plt.plot(snr_dB_list, CRB_UPGA_J10_decay, ':d', color='purple', linewidth=3, markersize=7, label=label_UPGA_J10_decay)
if run_UPGA_J20_decay == 1:
    plt.plot(snr_dB_list, CRB_UPGA_J20_decay, ':p', color='brown', linewidth=3, markersize=7, label=label_UPGA_J20_decay)
if run_UPGA_J_GradReuse == 1:
    plt.plot(snr_dB_list, CRB_UPGA_J_GradReuse, ':^', color='teal', linewidth=3, markersize=7, label=label_UPGA_J_GradReuse)
# if benchmark == 1:
#     plt.plot(snr_dB_list, CRB_SCA, '-x', color='black', linewidth=3, markersize=7, label=label_SCA)
#     plt.plot(snr_dB_list, CRB_ZF, '-o', color='purple', linewidth=3, markersize=7, label=label_ZF)

system_params = '$N=' + str(Nt) + ', M=' + str(M) + ', N_{\\mathrm{RF}}=' + str(Nrf) + ', \\omega=' + str(OMEGA) + '$'
# plt.title(system_params)
plt.xlabel('SNR [dB]')
plt.ylabel(r'$\mathrm{CRB}$')
plt.grid()
plt.legend(loc='upper right', labelspacing  = 0.15)
plt.savefig(directory_result + 'CRB_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png')
plt.savefig(directory_result + 'CRB_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps')


# Save SNR-curve data for MATLAB plotting (rate, CRB, and objective)
print('Saving SNR-curve results to .mat file...')
mat_data = {'snr_dB_list': snr_dB_list}

if run_conv_PGA == 1:
    mat_data['rate_conv_PGA_J1'] = rate_conv_PGA
    mat_data['crb_conv_PGA_J1'] = CRB_conv_PGA
    mat_data['obj_conv_PGA_J1'] = OMEGA * rate_conv_PGA + CRB_conv_PGA
if run_UPGA_J1 == 1:
    mat_data['rate_UPGA_J1'] = rate_UPGA_J1
    mat_data['crb_UPGA_J1'] = CRB_UPGA_J1
    mat_data['obj_UPGA_J1'] = OMEGA * rate_UPGA_J1 + CRB_UPGA_J1
if run_UPGA_J10 == 1:
    mat_data['rate_UPGA_J10'] = rate_UPGA_J10
    mat_data['crb_UPGA_J10'] = CRB_UPGA_J10
    mat_data['obj_UPGA_J10'] = OMEGA * rate_UPGA_J10 + CRB_UPGA_J10
if run_UPGA_J20 == 1:
    mat_data['rate_UPGA_J20'] = rate_UPGA_J20
    mat_data['crb_UPGA_J20'] = CRB_UPGA_J20
    mat_data['obj_UPGA_J20'] = OMEGA * rate_UPGA_J20 + CRB_UPGA_J20
if run_conv_PGA_J10 == 1:
    mat_data['rate_conv_PGA_J10'] = rate_conv_PGA_J10
    mat_data['crb_conv_PGA_J10'] = CRB_conv_PGA_J10
    mat_data['obj_conv_PGA_J10'] = OMEGA * rate_conv_PGA_J10 + CRB_conv_PGA_J10
if run_UPGA_J10_decay == 1:
    mat_data['rate_UPGA_J10_decay'] = rate_UPGA_J10_decay
    mat_data['crb_UPGA_J10_decay'] = CRB_UPGA_J10_decay
    mat_data['obj_UPGA_J10_decay'] = OMEGA * rate_UPGA_J10_decay + CRB_UPGA_J10_decay
if run_UPGA_J20_decay == 1:
    mat_data['rate_UPGA_J20_decay'] = rate_UPGA_J20_decay
    mat_data['crb_UPGA_J20_decay'] = CRB_UPGA_J20_decay
    mat_data['obj_UPGA_J20_decay'] = OMEGA * rate_UPGA_J20_decay + CRB_UPGA_J20_decay
if run_UPGA_J_GradReuse == 1:
    mat_data['rate_UPGA_J_GradReuse'] = rate_UPGA_J_GradReuse
    mat_data['crb_UPGA_J_GradReuse'] = CRB_UPGA_J_GradReuse
    mat_data['obj_UPGA_J_GradReuse'] = OMEGA * rate_UPGA_J_GradReuse + CRB_UPGA_J_GradReuse

mat_file_name = directory_result + 'snr_results_' + str(Nt) + '_' + str(OMEGA) + '.mat'
scipy.io.savemat(mat_file_name, mat_data)
print(f'  Saved to {mat_file_name}')



plt.show()
