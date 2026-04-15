from PGA_models import *
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
save_result = 0

# ///////////////////////////////////////// LOAD DATA ///////////////////////////////////
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, : test_size, :, :]

# ///////////////////////////////////////// RUN MODELS ///////////////////////////////////
if run_program == 1:

    # ============================== RKD MODEL (J=10) ==============================
    if run_RKD_Distillation == 1:
        print('Running UPGA J10 (RKD)...')

        model_UPGA_J10_RKD = PGA_Unfold_J10(step_size_UPGA_J10)
        model_UPGA_J10_RKD.load_state_dict(
            torch.load(directory_model + 'UPGA_J10_RKD.pth')
        )

        rate_UPGA_J10_RKD, tau_UPGA_J10_RKD, _, _ = model_UPGA_J10_RKD.execute_PGA(
            H_test, R, snr, n_iter_outer, n_iter_inner_J10
        )

        rate_iter_UPGA_J10_RKD = [
            r.detach().numpy() for r in (sum(rate_UPGA_J10_RKD) / len(H_test[0]))
        ]
        tau_iter_UPGA_J10_RKD = [
            e.detach().numpy() for e in (sum(tau_UPGA_J10_RKD) / len(H_test[0]))
        ]

    # ============================== UPGA J10 ==============================
    if run_UPGA_J10 == 1:
        print('Running unfolded PGA with J = 10...')

        model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
        model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10))

        rate_UPGA_J10, tau_UPGA_J10, _, _ = model_UPGA_J10.execute_PGA(
            H_test, R, snr, n_iter_outer, n_iter_inner_J10
        )

        rate_iter_UPGA_J10 = [
            r.detach().numpy() for r in (sum(rate_UPGA_J10) / len(H_test[0]))
        ]
        tau_iter_UPGA_J10 = [
            e.detach().numpy() for e in (sum(tau_UPGA_J10) / len(H_test[0]))
        ]

    # ============================== UPGA J20 ==============================
    if run_UPGA_J20 == 1:
        print('Running unfolded PGA with J = 20...')

        model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)

        rate_UPGA_J20, tau_UPGA_J20, _, _ = model_UPGA_J20.execute_PGA(
            H_test, R, snr, n_iter_outer, n_iter_inner_J20
        )

        rate_iter_UPGA_J20 = [
            r.detach().numpy() for r in (sum(rate_UPGA_J20) / len(H_test[0]))
        ]
        tau_iter_UPGA_J20 = [
            e.detach().numpy() for e in (sum(tau_UPGA_J20) / len(H_test[0]))
        ]


# ///////////////////////////////////////// PLOTTING ///////////////////////////////////
if plot_figure == 1:

    print('Plotting figures...')

    iter_number = np.array(list(range(n_iter_outer + 1)))

    # ===================== OBJECTIVE =====================
    fig_obj = plt.figure(5)

    if run_UPGA_J10 == 1:
        obj_J10 = [r - OMEGA * t for r, t in zip(rate_iter_UPGA_J10, tau_iter_UPGA_J10)]
        plt.plot(iter_number, obj_J10, ':*', markevery=5,
                 color='blue', linewidth=3, markersize=7,
                 label='Unfolded PGA (J=10)')

    if run_UPGA_J20 == 1:
        obj_J20 = [r - OMEGA * t for r, t in zip(rate_iter_UPGA_J20, tau_iter_UPGA_J20)]
        plt.plot(iter_number, obj_J20, '-', markevery=5,
                 color='red', linewidth=3, markersize=7,
                 label='Unfolded PGA (J=20)')

    if run_RKD_Distillation == 1:
        obj_RKD = [r - OMEGA * t for r, t in zip(rate_iter_UPGA_J10_RKD, tau_iter_UPGA_J10_RKD)]
        plt.plot(iter_number, obj_RKD, '-.', markevery=5,
                 color='green', linewidth=3, markersize=7,
                 label='UPGA (J=10, RKD)')

    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'$R - \omega \bar{\tau}$', fontsize=14)
    plt.grid()
    plt.legend()

    plt.savefig(directory_result + f'objective_vs_iter_{Nt}_{OMEGA}.png')
    plt.savefig(directory_result + f'objective_vs_iter_{Nt}_{OMEGA}.eps')


    # ===================== TRADEOFF =====================
    fig_tradeoff = plt.figure(3)

    if run_UPGA_J10 == 1:
        plt.plot(tau_iter_UPGA_J10, rate_iter_UPGA_J10, ':*', markevery=5,
                 color='blue', linewidth=3, markersize=7,
                 label='Unfolded PGA (J=10)')

    if run_UPGA_J20 == 1:
        plt.plot(tau_iter_UPGA_J20, rate_iter_UPGA_J20, '-', markevery=5,
                 color='red', linewidth=3, markersize=7,
                 label='Unfolded PGA (J=20)')

    if run_RKD_Distillation == 1:
        plt.plot(tau_iter_UPGA_J10_RKD, rate_iter_UPGA_J10_RKD, '-.', markevery=5,
                 color='green', linewidth=3, markersize=7,
                 label='UPGA (J=10, RKD)')

    plt.xlabel(r'$\bar{\tau}$', fontsize=14)
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
    plt.grid()
    plt.legend()

    plt.savefig(directory_result + f'tradeoff_vs_iter_{Nt}_{OMEGA}.png')
    plt.savefig(directory_result + f'tradeoff_vs_iter_{Nt}_{OMEGA}.eps')


    # ===================== BEAM ERROR =====================
    fig_tau = plt.figure(2)

    if run_UPGA_J10 == 1:
        plt.plot(iter_number, tau_iter_UPGA_J10, ':*', markevery=5,
                 color='blue', linewidth=3, markersize=7,
                 label='Unfolded PGA (J=10)')

    if run_UPGA_J20 == 1:
        plt.plot(iter_number, tau_iter_UPGA_J20, '-', markevery=5,
                 color='red', linewidth=3, markersize=7,
                 label='Unfolded PGA (J=20)')

    if run_RKD_Distillation == 1:
        plt.plot(iter_number, tau_iter_UPGA_J10_RKD, '-.', markevery=5,
                 color='green', linewidth=3, markersize=7,
                 label='UPGA (J=10, RKD)')

    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'$\bar{\tau}$', fontsize=14)
    plt.grid()
    plt.legend()

    plt.savefig(directory_result + f'beampattern_error_vs_iter_{Nt}_{OMEGA}.png')
    plt.savefig(directory_result + f'beampattern_error_vs_iter_{Nt}_{OMEGA}.eps')

plt.show()