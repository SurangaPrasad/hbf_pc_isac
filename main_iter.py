from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
save_result = 0
# torch.manual_seed(3407)
# ///////////////////////////////////////// SHOW OBJECTIVE VALUES OVER ITERATIONS ///////////////////////////////////
# Load training data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_train[:, :test_size, :, :]
# H_test = H_train[:, 100:1+100, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, : test_size, :, :]

if run_program == 1:
    # ====================================================== Conv. PGA ====================================
    if run_conv_PGA == 1:
        print('Running conventional PGA...')
        model_conv_PGA = PGA_Conv(step_size_conv_PGA)
        rate_conv, tau_conv, F_conv, W_conv = model_conv_PGA.execute_PGA(H_test, R, snr, n_iter_outer)
        rate_iter_conv = [r.detach().cpu().numpy() for r in (sum(rate_conv) / len(H_test[0]))]
        tau_iter_conv = [e.detach().cpu().numpy() for e in (sum(tau_conv) / (len(H_test[0])))]
    # ====================================================== Conv. PGA with J = 10 ====================================
    if run_conv_PGA_J10 == 1:
        print('Running conventional PGA with J = 10...')
        model_conv_PGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
        rate_conv_PGA_J10, crb_conv_PGA_J10, power_conv_PGA_J10, F_conv_PGA_J10, W_conv_PGA_J10 = model_conv_PGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J10)
        # rate_conv_PGA_J10: (B, n_outer*(J+1))  — average over batch
        rate_iter_conv_PGA_J10  = rate_conv_PGA_J10.mean(0).cpu().numpy()
        crb_iter_conv_PGA_J10   = crb_conv_PGA_J10.mean(0).cpu().numpy()
        power_iter_conv_PGA_J10 = power_conv_PGA_J10.mean(0).cpu().numpy()

    # ====================================================== Conv. PGA with J = 20 ====================================
    if run_conv_PGA_J20 == 1:
        print('Running conventional PGA with J = 20...')
        model_conv_PGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
        rate_conv_PGA_J20, crb_conv_PGA_J20, power_conv_PGA_J20, F_conv_PGA_J20, W_conv_PGA_J20 = model_conv_PGA_J20.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J20)
        rate_iter_conv_PGA_J20 = rate_conv_PGA_J20.mean(0).cpu().numpy()
        crb_iter_conv_PGA_J20  = crb_conv_PGA_J20.mean(0).cpu().numpy()
        power_iter_conv_PGA_J20 = power_conv_PGA_J20.mean(0).cpu().numpy()
    # ====================================================== Unfolded PGA with J = 1====================================
    if run_UPGA_J1 == 1:
        print('Running unfolded PGA with J = 1...')
        # Create new model and load states
        model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
        model_UPGA_J1.load_state_dict(torch.load(model_file_name_UPGA_J1, map_location=device))

        # executing unfolded PGA on the test set
        sum_rate_UPGA_J1, tau_UPGA_J1, F_UPGA_J1, W_UPGA_J1 = model_UPGA_J1.execute_PGA(H_test, R, snr, n_iter_outer)
        rate_iter_UPGA_J1 = [r.detach().cpu().numpy() for r in (sum(sum_rate_UPGA_J1) / len(H_test[0]))]
        tau_iter_UPGA_J1 = [e.detach().cpu().numpy() for e in (sum(tau_UPGA_J1) / (len(H_test[0])))]

    # ====================================================== Proposed Unfolded PGA light ====================================
    if run_UPGA_J10 == 1:
        print('Running unfolded PGA with J = 10...')
        # Create new model and load states
        model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
        model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10, map_location=device))

        sum_rate_UPGA_J10, crb_UPGA_J10, power_UPGA_J10, F_UPGA_J10, W_UPGA_J10 = model_UPGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                            n_iter_inner_J10)
        print(f'Shape of the sum_rate_UPGA_J10: {sum_rate_UPGA_J10.shape}')
        rate_iter_UPGA_J10  = sum_rate_UPGA_J10.mean(0).cpu().numpy()
        crb_iter_UPGA_J10   = crb_UPGA_J10.mean(0).cpu().numpy()
        power_iter_UPGA_J10 = power_UPGA_J10.mean(0).cpu().numpy()

    # ====================================================== Proposed Unfolded PGA ====================================
    if run_UPGA_J20 == 1:
        print('Running unfolded PGA with J = 20...')
        # Create new model and load states
        model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
        model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))

        sum_rate_UPGA_J20, crb_UPGA_J20,power_UPGA_J20, F_UPGA_J20, W_UPGA_J20 = model_UPGA_J20.execute_PGA(H_test, xi_0, A_dot, R_N_inv, snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J20)
        rate_iter_UPGA_J20 = sum_rate_UPGA_J20.mean(0).cpu().numpy()
        crb_iter_UPGA_J20  = crb_UPGA_J20.mean(0).cpu().numpy()
        power_iter_UPGA_J20 = power_UPGA_J20.mean(0).cpu().numpy()
    
    # ====================================================== Propsed Unofolded PGA with PRCDN ====================================

    if run_UPGA_J10_PRCDN:
        print('Running unfolded PGA with J = 10 and PRCDN...')
        # Create new model and load states
        model_UPGA_J10_PRCDN = PGA_Unfold_J10_PRCDN(n_iter_inner_J10, n_iter_outer, dim_F=64, dim_W=4)
        model_UPGA_J10_PRCDN.load_state_dict(torch.load(model_file_name_UPGA_J10_PRCDN, map_location=device))

        sum_rate_UPGA_J10_PRCDN, crb_UPGA_J10_PRCDN, power_UPGA_J10_PRCDN, F_UPGA_J10_PRCDN, W_UPGA_J10_PRCDN = model_UPGA_J10_PRCDN.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J10)
        rate_iter_UPGA_J10_PRCDN  = sum_rate_UPGA_J10_PRCDN.mean(0).cpu().numpy()
        crb_iter_UPGA_J10_PRCDN   = crb_UPGA_J10_PRCDN.mean(0).cpu().numpy()
        power_iter_UPGA_J10_PRCDN = power_UPGA_J10_PRCDN.mean(0).cpu().numpy()
    # ====================================================== Proposed Unfolded PGA with decaying J ====================================
    if run_UPGA_J10_decay == 1:
        print('Running unfolded PGA with decaying J...')
        model_UPGA_J10_decay = PGA_Unfold_J10_decay(step_size_UPGA_J10_decay)
        model_UPGA_J10_decay.load_state_dict(torch.load(model_file_name_UPGA_J10_decay, map_location=device))

        sum_rate_UPGA_J10_decay, crb_UPGA_J10_decay, power_UPGA_J10_decay, F_UPGA_J10_decay, W_UPGA_J10_decay = model_UPGA_J10_decay.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                            n_iter_inner_J10)
        rate_iter_UPGA_J10_decay  = sum_rate_UPGA_J10_decay.mean(0).cpu().numpy()
        crb_iter_UPGA_J10_decay   = crb_UPGA_J10_decay.mean(0).cpu().numpy()
        power_iter_UPGA_J10_decay = power_UPGA_J10_decay.mean(0).cpu().numpy()
        inner_iter_history_UPGA_J10_decay = list(model_UPGA_J10_decay.inner_iter_history)
    # ====================================================== Proposed Unfolded PGA with decaying J (J_max=20) =================
    if run_UPGA_J20_decay == 1:
        print('Running unfolded PGA with decaying J (J_max=20)...')
        model_UPGA_J20_decay = PGA_Unfold_J20_decay(step_size_UPGA_J20_decay)
        # model_UPGA_J20_decay.load_state_dict(torch.load(model_file_name_UPGA_J20_decay, map_location=device))

        sum_rate_UPGA_J20_decay, crb_UPGA_J20_decay, power_UPGA_J20_decay, F_UPGA_J20_decay, W_UPGA_J20_decay = model_UPGA_J20_decay.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                            n_iter_inner_J20)
        rate_iter_UPGA_J20_decay  = sum_rate_UPGA_J20_decay.mean(0).cpu().numpy()
        crb_iter_UPGA_J20_decay   = crb_UPGA_J20_decay.mean(0).cpu().numpy()
        power_iter_UPGA_J20_decay = power_UPGA_J20_decay.mean(0).cpu().numpy()
        inner_iter_history_UPGA_J20_decay = list(model_UPGA_J20_decay.inner_iter_history)
    # ====================================================== Proposed Unfolded PGA with gradient reuse ====================================
    if run_UPGA_J_GradReuse == 1:
        print('Running unfolded PGA with gradient reuse (J = 10)...')
        model_UPGA_J_GradReuse = PGA_Unfold_J_GradReuse(step_size_UPGA_J_GradReuse)
        # model_UPGA_J_GradReuse.load_state_dict(torch.load(model_file_name_UPGA_J10, map_location=device))

        sum_rate_UPGA_J_GradReuse, crb_UPGA_J_GradReuse, power_UPGA_J_GradReuse, F_UPGA_J_GradReuse, W_UPGA_J_GradReuse = model_UPGA_J_GradReuse.execute_PGA(
            H_test, xi_0, A_dot, R_N_inv, snr, n_iter_outer, n_iter_inner_J10)
        print(f'  GradReuse fallback recomputations: {model_UPGA_J_GradReuse.grad_recalc_count}')
        rate_iter_UPGA_J_GradReuse  = sum_rate_UPGA_J_GradReuse.mean(0).cpu().numpy()
        crb_iter_UPGA_J_GradReuse   = crb_UPGA_J_GradReuse.mean(0).cpu().numpy()
        power_iter_UPGA_J_GradReuse = power_UPGA_J_GradReuse.mean(0).cpu().numpy()
    # ====================================================== Propsed Unofolded PGA with RMSProp-like adaptive step sizes ====================================
    if run_UPGA_J10_RMSProp == 1:
        print('Running unfolded PGA with J = 10 and RMSProp-like adaptive step sizes...')
        # Create new model and load states
        model_UPGA_J10_RMSProp = PGA_Unfold_J10_RMSProp()
        # model_UPGA_J10_RMSProp.load_state_dict(torch.load(model_file_name_UPGA_J10_RMSProp))

        sum_rate_UPGA_J10_RMSProp, crb_UPGA_J10_RMSProp, F_UPGA_J10_RMSProp, W_UPGA_J10_RMSProp = model_UPGA_J10_RMSProp.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J10)
        rate_iter_UPGA_J10_RMSProp = sum_rate_UPGA_J10_RMSProp.mean(0).cpu().numpy()
        crb_iter_UPGA_J10_RMSProp  = crb_UPGA_J10_RMSProp.mean(0).cpu().numpy()

    # ============================== generate beampattern ////////////////////////////////////////////////////////////////////
    print('generating beampattern...')
    if run_conv_PGA == 1:
        beam_conv_PGA = get_beampattern(F_conv, W_conv, at, snr)
    if run_UPGA_J1 == 1:
        beam_UPGA_J1 = get_beampattern(F_UPGA_J1, W_UPGA_J1, at, snr)
    if run_UPGA_J10 == 1:
        beam_UPGA_J10 = get_beampattern(F_UPGA_J10, W_UPGA_J10, at, snr)
    if run_UPGA_J20 == 1:
        beam_UPGA_J20 = get_beampattern(F_UPGA_J20, W_UPGA_J20, at, snr)
    if run_conv_PGA_J10 == 1:
        beam_conv_PGA_J10 = get_beampattern(F_conv_PGA_J10, W_conv_PGA_J10, at, snr)

    # ////////////////////////////////////////////////////////////////////////////////////////////
    #                                SAVE RESULTS
    # //////////////////////////////////////////////////////////////////////////////////////////////
    if save_result == 1:
        print('Saving results...')
        if run_conv_PGA == 1:
            result_conv_PGA_file_name = directory_result + 'result_vs_iter_conv.npz'
            np.savez(result_conv_PGA_file_name, name1=rate_iter_conv, name2=tau_iter_conv, name3=beam_conv_PGA)
        if run_UPGA_J1 == 1:
            result_UPGA_J1_file_name = directory_result + 'result_vs_iter_UPGA_J1.npz'
            np.savez(result_UPGA_J1_file_name, name1=rate_iter_UPGA_J1, name2=tau_iter_UPGA_J1, name3=beam_UPGA_J1)
        if run_UPGA_J10 == 1:
            result_UPGA_J10_file_name = directory_result + 'result_vs_iter_UPGA_J10.npz'
            np.savez(result_UPGA_J10_file_name, name1=rate_iter_UPGA_J10, name2=crb_iter_UPGA_J10, name3=beam_UPGA_J10)
        if run_UPGA_J20 == 1:
            result_UPGA_J20_file_name = directory_result + 'result_vs_iter_UPGA_J20.npz'
            np.savez(result_UPGA_J20_file_name, name1=rate_iter_UPGA_J20, name2=crb_iter_UPGA_J20, name3=beam_UPGA_J20)

if plot_figure == 1:

    # ///////////////////////////////////////// SHOW OBJECTIVE VALUES OVER ITERATIONS ///////////////////////////////////
    benchmark = 0
    iter_number_conv_PGA = np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_J1  = np.array(list(range(n_iter_outer + 1)))
    # For J-inner models: flattened x-axis, length = n_outer * (J+1)
    # Each outer step ii contributes slots [ii*(J+1)+0 .. ii*(J+1)+J]
    iter_number_UPGA_J10 = np.arange(n_iter_outer * (n_iter_inner_J10 + 1))
    iter_number_UPGA_J20 = np.arange(n_iter_outer * (n_iter_inner_J20 + 1))
    # Fractional x-axis: map each slot back to a real-valued outer iteration
    # slot ii*(J+1)+0 → ii (after W-update), slot ii*(J+1)+jj+1 → ii + (jj+1)/(J+1)
    def fractional_iters(n_outer, n_inner):
        x = []
        for ii in range(n_outer):
            x.append(float(ii))                            # W-update slot
            for jj in range(n_inner):
                x.append(ii + (jj + 1) / (n_inner + 1))   # inner F-update slots
        return np.array(x)
    frac_J10 = fractional_iters(n_iter_outer, n_iter_inner_J10)
    frac_J20 = fractional_iters(n_iter_outer, n_iter_inner_J20)

    def fractional_iters_variable(inner_iter_history):
        """Fractional x-axis for variable inner-iteration schedules."""
        x = []
        for ii, n_inner_ii in enumerate(inner_iter_history):
            x.append(float(ii))
            for jj in range(n_inner_ii):
                x.append(ii + (jj + 1) / (n_inner_ii + 1))
        return np.array(x)
    if run_UPGA_J10_decay == 1:
        frac_J10_decay = fractional_iters_variable(inner_iter_history_UPGA_J10_decay)
    else:
        frac_J10_decay = np.array([])
    if run_UPGA_J20_decay == 1:
        frac_J20_decay_var = fractional_iters_variable(inner_iter_history_UPGA_J20_decay)
    else:
        frac_J20_decay_var = np.array([])
    # Indices of the last inner step of each outer iteration in the flattened arrays
    # J=10: indices 10, 21, 32, ...  (block size J+1=11, last slot = J=10)
    # J=20: indices 20, 41, 62, ...  (block size J+1=21, last slot = J=20)
    outer_idx_J10 = np.arange(n_iter_inner_J10,
                              n_iter_outer * (n_iter_inner_J10 + 1),
                              n_iter_inner_J10 + 1)   # length = n_iter_outer
    outer_idx_J20 = np.arange(n_iter_inner_J20,
                              n_iter_outer * (n_iter_inner_J20 + 1),
                              n_iter_inner_J20 + 1)   # length = n_iter_outer
    # outer_idx for J_decay/adaptive schedule: W-update is the LAST slot of each block
    if run_UPGA_J10_decay == 1:
        outer_idx_J10_decay = []
        _pos = 0
        for _ni in inner_iter_history_UPGA_J10_decay:
            _pos += _ni
            outer_idx_J10_decay.append(_pos)
            _pos += 1
        outer_idx_J10_decay = np.array(outer_idx_J10_decay)
        iter_outer_x_J10_decay = np.arange(1, len(outer_idx_J10_decay) + 1)
    else:
        outer_idx_J10_decay = np.array([])
        iter_outer_x_J10_decay = np.array([])
    if run_UPGA_J20_decay == 1:
        outer_idx_J20_decay = []
        _pos = 0
        for _ni in inner_iter_history_UPGA_J20_decay:
            _pos += _ni
            outer_idx_J20_decay.append(_pos)
            _pos += 1
        outer_idx_J20_decay = np.array(outer_idx_J20_decay)
        iter_outer_x_J20_decay = np.arange(1, len(outer_idx_J20_decay) + 1)
    else:
        outer_idx_J20_decay = np.array([])
        iter_outer_x_J20_decay = np.array([])
    # J_GradReuse has the same fixed J=10 structure as J10
    outer_idx_J_GradReuse = outer_idx_J10
    frac_J_GradReuse = frac_J10
    iter_outer_x  = np.arange(1, n_iter_outer + 1)    # x-axis: 1 .. n_iter_outer

    # //////////////////////////////// LOADING RESULTS //////////////////////////////////////////
    if save_result == 1:
        if run_conv_PGA == 1:
            result_file_name = directory_result + 'result_vs_iter_conv.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']
        if run_UPGA_J1 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J1.npz'
            result = np.load(result_file_name)
            rate_iter_UPGA_J1, tau_iter_UPGA_J1, beam_UPGA_J1 = result['name1'], result['name2'], result['name3']
        if run_UPGA_J10 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J10.npz'
            result = np.load(result_file_name)
            rate_iter_UPGA_J10, tau_iter_UPGA_J10, beam_UPGA_J10 = result['name1'], result['name2'], result['name3']
        if run_UPGA_J20 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J20.npz'
            result = np.load(result_file_name)
            rate_iter_UPGA_J20, tau_iter_UPGA_J20, beam_UPGA_J20 = result['name1'], result['name2'], result['name3']

    #  /////////////////////////////////////////////////////////////////////////////////////////
    #                               PLOT FIGURES
    # //////////////////////////////////////////////////////////////////////////////////////////
    print('Plotting figures...')
    system_params = (
        rf'$N={Nt}, M={M}, N_{{\mathrm{{RF}}}}={Nrf}, '
        rf'\mathrm{{SNR}}={snr_dB} \mathrm{{dB}}, '
        rf'\omega={OMEGA}$'
    )

    # load benchmark results
    if benchmark == 1:
        benchmark_results = scipy.io.loadmat(directory_benchmark + 'result_benchmark')
        rate_ZF = np.squeeze(benchmark_results['rate_ZF_mean'])
        rate_SCA = np.squeeze(benchmark_results['rate_SCA_mean'])
        tau_ZF = np.squeeze(benchmark_results['tau_ZF_mean'])
        tau_SCA = np.squeeze(benchmark_results['tau_SCA_mean'])

        idx_snr = np.where(snr_dB_list == snr_dB)
        rate_ZF = rate_ZF[idx_snr] * np.ones(n_iter_outer + 1)
        rate_SCA = rate_SCA[idx_snr] * np.ones(n_iter_outer + 1)
        tau_ZF = tau_ZF[idx_snr] * np.ones(n_iter_outer + 1)
        tau_SCA = tau_SCA[idx_snr] * np.ones(n_iter_outer + 1)

        beam_ZF = np.squeeze(benchmark_results['beam_ZF_mean'][:, idx_snr])
        beam_SCA = np.squeeze(benchmark_results['beam_SCA_mean'][:, idx_snr])



    # ==================================== RATES (outer iters only) ================================================
    plt.figure()
    if run_UPGA_J1 == 1:
        plt.plot(iter_number_UPGA_J1, rate_iter_UPGA_J1, '--', markevery=5, color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
    if run_UPGA_J10 == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J10[outer_idx_J10], ':*', markevery=5, color='red', linewidth=3, markersize=7,
                 label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J20[outer_idx_J20], '-', markevery=5, color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
    if run_conv_PGA == 1:
        plt.plot(iter_number_conv_PGA, rate_iter_conv, ':', markevery=5, color='black', linewidth=3, markersize=7, label=label_conv)
    if benchmark == 1:
        plt.plot(iter_number_conv_PGA, rate_SCA, '-x', markevery=5, color='black', linewidth=3, markersize=7, label=label_SCA)
        plt.plot(iter_number_conv_PGA, rate_ZF, '-o', markevery=5, color='purple', linewidth=3, markersize=7, label=label_ZF)
    if run_conv_PGA_J10 == 1:
        plt.plot(iter_outer_x, rate_iter_conv_PGA_J10[outer_idx_J10], ':*', markevery=5, color='orange', linewidth=3, markersize=7, label='PGA (J=10)')
    if run_conv_PGA_J20 ==1:
        plt.plot(iter_outer_x, rate_iter_conv_PGA_J20[outer_idx_J20], ':s', markevery=5, color='black', linewidth=3, markersize=7, label='PGA (J=20)')    
    if run_UPGA_J10_PRCDN == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J10_PRCDN[outer_idx_J10], ':*', markevery=5, color='green', linewidth=3, markersize=7, label='PGA (J=10, PRCDN)')
    if run_UPGA_J10_decay == 1:
        plt.plot(iter_outer_x_J10_decay, rate_iter_UPGA_J10_decay[outer_idx_J10_decay], ':d', markevery=5, color='purple', linewidth=3, markersize=7, label='PGA (J=10, decay)')
    if run_UPGA_J20_decay == 1:
        plt.plot(iter_outer_x_J20_decay, rate_iter_UPGA_J20_decay[outer_idx_J20_decay], ':p', markevery=5, color='brown', linewidth=3, markersize=7, label='PGA (J=20, decay)')
    if run_UPGA_J_GradReuse == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse], ':^', markevery=5, color='teal', linewidth=3, markersize=7, label='PGA (J=10, GradReuse)')
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel('$R$ [bits/s/Hz]', fontsize="14")
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing  = 0.15)
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # ==================================== CRB (outer iters only) ================================================
    plt.figure()
    if run_conv_PGA_J10 == 1:
        plt.plot(iter_outer_x, crb_iter_conv_PGA_J10[outer_idx_J10], ':*', markevery=5, color='orange', linewidth=3, markersize=7, label='PGA (J=10)')
    if run_conv_PGA_J20 == 1:
        plt.plot(iter_outer_x, crb_iter_conv_PGA_J20[outer_idx_J20], ':s', markevery=5, color='black', linewidth=3, markersize=7, label='PGA (J=20)')
    if run_UPGA_J10 == 1:
        plt.plot(iter_outer_x, crb_iter_UPGA_J10[outer_idx_J10], ':*', markevery=5, color='blue', linewidth=3, markersize=7, label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        plt.plot(iter_outer_x, crb_iter_UPGA_J20[outer_idx_J20], ':s', markevery=5, color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
    if run_UPGA_J10_RMSProp == 1:
        plt.plot(iter_outer_x, crb_iter_UPGA_J10_RMSProp[outer_idx_J10], ':', markevery=5, color='green', linewidth=3, markersize=7, label='PGA (J=10, RMSProp)')
    if run_UPGA_J10_PRCDN == 1:
        plt.plot(iter_outer_x, crb_iter_UPGA_J10_PRCDN[outer_idx_J10], ':*', markevery=5, color='green', linewidth=3, markersize=7, label='PGA (J=10, PRCDN)')
    if run_UPGA_J10_decay == 1:
        plt.plot(iter_outer_x_J10_decay, crb_iter_UPGA_J10_decay[outer_idx_J10_decay], ':d', markevery=5, color='purple', linewidth=3, markersize=7, label='PGA (J=10, decay)')
    if run_UPGA_J20_decay == 1:
        plt.plot(iter_outer_x_J20_decay, crb_iter_UPGA_J20_decay[outer_idx_J20_decay], ':p', markevery=5, color='brown', linewidth=3, markersize=7, label='PGA (J=20, decay)')
    if run_UPGA_J_GradReuse == 1:
        plt.plot(iter_outer_x, crb_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse], ':^', markevery=5, color='teal', linewidth=3, markersize=7, label='PGA (J=10, GradReuse)')
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel(r'$1/\text{crb}$', fontsize="14")
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing  = 0.15)
    plt.savefig(directory_result + 'crb_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'crb_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # ===================== OBJECTIVE (outer iters only) =============================================
    plt.figure()
    fig_obj = plt.figure(5)
    if run_UPGA_J1 == 1:
        obj_iter_UPGA_J1 = [rate - OMEGA * tau for rate, tau in zip(rate_iter_UPGA_J1, tau_iter_UPGA_J1)]
        plt.plot(iter_number_UPGA_J1, obj_iter_UPGA_J1, '--', markevery=5, color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
    if run_conv_PGA_J10 == 1:
        obj_iter_conv_PGA_J10 = OMEGA * rate_iter_conv_PGA_J10[outer_idx_J10] + crb_iter_conv_PGA_J10[outer_idx_J10]
        plt.plot(iter_outer_x, obj_iter_conv_PGA_J10, ':*', markevery=5, color='orange', linewidth=3, markersize=7, label='PGA (J=10)')
    if run_conv_PGA_J20 == 1:
        obj_iter_conv_PGA_J20 = OMEGA * rate_iter_conv_PGA_J20[outer_idx_J20] + crb_iter_conv_PGA_J20[outer_idx_J20]
        plt.plot(iter_outer_x, obj_iter_conv_PGA_J20, '-', markevery=5, color='black', linewidth=3, markersize=7, label='PGA (J=20)')
    if run_UPGA_J10 == 1:
        obj_iter_UPGA_J10 = OMEGA * rate_iter_UPGA_J10[outer_idx_J10] + crb_iter_UPGA_J10[outer_idx_J10]
        plt.plot(iter_outer_x, obj_iter_UPGA_J10, ':*', markevery=5, color='blue', linewidth=3, markersize=7, label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        obj_iter_UPGA_J20 = OMEGA * rate_iter_UPGA_J20[outer_idx_J20] + crb_iter_UPGA_J20[outer_idx_J20]
        plt.plot(iter_outer_x, obj_iter_UPGA_J20, '-', markevery=5, color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
    if run_UPGA_J10_RMSProp == 1:
        obj_iter_UPGA_J10_RMSProp = OMEGA * rate_iter_UPGA_J10_RMSProp[outer_idx_J10] + crb_iter_UPGA_J10_RMSProp[outer_idx_J10]
        plt.plot(iter_outer_x, obj_iter_UPGA_J10_RMSProp, ':', markevery=5, color='green', linewidth=3, markersize=7, label='PGA (J=10, RMSProp)')
    if run_UPGA_J10_PRCDN == 1:
        obj_iter_UPGA_J10_PRCDN = OMEGA * rate_iter_UPGA_J10_PRCDN[outer_idx_J10] + crb_iter_UPGA_J10_PRCDN[outer_idx_J10]
        plt.plot(iter_outer_x, obj_iter_UPGA_J10_PRCDN, ':*', markevery=5, color='green', linewidth=3, markersize=7, label='PGA (J=10, PRCDN)')
    if run_UPGA_J10_decay == 1:
        obj_iter_UPGA_J10_decay = OMEGA * rate_iter_UPGA_J10_decay[outer_idx_J10_decay] + crb_iter_UPGA_J10_decay[outer_idx_J10_decay]
        plt.plot(iter_outer_x_J10_decay, obj_iter_UPGA_J10_decay, ':d', markevery=5, color='purple', linewidth=3, markersize=7, label='PGA (J=10, decay)')
    if run_UPGA_J20_decay == 1:
        obj_iter_UPGA_J20_decay = OMEGA * rate_iter_UPGA_J20_decay[outer_idx_J20_decay] + crb_iter_UPGA_J20_decay[outer_idx_J20_decay]
        plt.plot(iter_outer_x_J20_decay, obj_iter_UPGA_J20_decay, ':p', markevery=5, color='brown', linewidth=3, markersize=7, label='PGA (J=20, decay)')
    if run_UPGA_J_GradReuse == 1:
        obj_iter_UPGA_J_GradReuse = OMEGA * rate_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse] + crb_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse]
        plt.plot(iter_outer_x, obj_iter_UPGA_J_GradReuse, ':^', markevery=5, color='teal', linewidth=3, markersize=7, label=label_UPGA_J_GradReuse)
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel(r'$\omega R + 1/\text{crb}$', fontsize="14")
    plt.title("Objective function vs Iterations", fontsize="14")
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing  = 0.15)
    plt.savefig(directory_result + 'objective_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'objective_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')


    # ===================== OBJECTIVE INCLUDING ALL INNER ITERATIONS (first 20 outer iters) =========
    # x-axis: fractional outer iteration so inner steps are visible between integers
    n_plot_outer = 40   # number of outer iterations to display
    mask_J10 = frac_J10 < n_plot_outer
    mask_J20 = frac_J20 < n_plot_outer
    mask_J10_decay = frac_J10_decay < n_plot_outer
    fig_obj_inner = plt.figure(6)
    if run_conv_PGA_J10 == 1:
        obj = OMEGA * rate_iter_conv_PGA_J10 + crb_iter_conv_PGA_J10
        plt.plot(frac_J10[mask_J10], obj[mask_J10], ':*', markevery=10, color='orange', linewidth=2, markersize=5, label='PGA (J=10)')
    if run_conv_PGA_J20 == 1:
        obj = OMEGA * rate_iter_conv_PGA_J20 + crb_iter_conv_PGA_J20
        plt.plot(frac_J20[mask_J20], obj[mask_J20], '-', markevery=10, color='black', linewidth=2, markersize=5, label='PGA (J=20)')
    if run_UPGA_J10 == 1:
        obj = OMEGA * rate_iter_UPGA_J10 + crb_iter_UPGA_J10
        plt.plot(frac_J10[mask_J10], obj[mask_J10], ':*', markevery=10, color='blue', linewidth=2, markersize=5, label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        obj = OMEGA * rate_iter_UPGA_J20 + crb_iter_UPGA_J20
        plt.plot(frac_J20[mask_J20], obj[mask_J20], '-', markevery=10, color='red', linewidth=2, markersize=5, label=label_UPGA_J20)
    if run_UPGA_J10_RMSProp == 1:
        obj = OMEGA * rate_iter_UPGA_J10_RMSProp + crb_iter_UPGA_J10_RMSProp
        plt.plot(frac_J10[mask_J10], obj[mask_J10], ':', markevery=10, color='purple', linewidth=2, markersize=5, label='PGA (J=10, RMSProp)')
    if run_UPGA_J10_PRCDN == 1:
        obj = OMEGA * rate_iter_UPGA_J10_PRCDN + crb_iter_UPGA_J10_PRCDN
        plt.plot(frac_J10[mask_J10], obj[mask_J10], ':s', markevery=10, color='green', linewidth=2, markersize=5, label='PGA (J=10, PRCDN)')
    if run_UPGA_J10_decay == 1:
        obj = OMEGA * rate_iter_UPGA_J10_decay + crb_iter_UPGA_J10_decay
        plt.plot(frac_J10_decay[mask_J10_decay], obj[mask_J10_decay], ':d', markevery=10, color='purple', linewidth=2, markersize=5, label=label_UPGA_J10_decay)
    if run_UPGA_J20_decay == 1:
        mask_J20_decay = frac_J20_decay_var < n_plot_outer
        obj = OMEGA * rate_iter_UPGA_J20_decay + crb_iter_UPGA_J20_decay
        plt.plot(frac_J20_decay_var[mask_J20_decay], obj[mask_J20_decay], ':p', markevery=10, color='brown', linewidth=2, markersize=5, label=label_UPGA_J20_decay)
    if run_UPGA_J_GradReuse == 1:
        mask_J_GradReuse = frac_J_GradReuse < n_plot_outer
        obj = OMEGA * rate_iter_UPGA_J_GradReuse + crb_iter_UPGA_J_GradReuse
        plt.plot(frac_J_GradReuse[mask_J_GradReuse], obj[mask_J_GradReuse], ':^', markevery=10, color='teal', linewidth=2, markersize=5, label='PGA (J=10, GradReuse)')
    # Mark outer-iteration boundaries with vertical grid lines
    for ii in range(1, n_plot_outer):
        plt.axvline(x=ii, color='grey', linestyle='--', linewidth=0.6, alpha=0.5)
    plt.xlabel(r'Outer iteration $I$ (inner steps shown as fractions)', fontsize="13")
    plt.ylabel(r'$\omega R + 1/\mathrm{CRB}$', fontsize="13")
    plt.title(f"Objective function — first {n_plot_outer} outer iterations (incl. inner)", fontsize="13")
    plt.grid(axis='y')
    plt.legend(loc='best', fontsize="12", labelspacing=0.15)
    plt.tight_layout()
    plt.savefig(directory_result + 'objective_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'objective_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.eps')


    # ===================== TRANSMIT POWER INCLUDING ALL INNER ITERATIONS (first 20 outer iters) =========
    # Only J10-based models return power_fes; J20 and RMSProp do not.
    fig_power_inner = plt.figure(7)
    if run_conv_PGA_J10 == 1:
        plt.plot(frac_J10[mask_J10], power_iter_conv_PGA_J10[mask_J10], ':*', markevery=10, color='orange', linewidth=2, markersize=5, label='PGA (J=10)')
    if run_UPGA_J10 == 1:
        plt.plot(frac_J10[mask_J10], power_iter_UPGA_J10[mask_J10], ':*', markevery=10, color='blue', linewidth=2, markersize=5, label=label_UPGA_J10)
    if run_UPGA_J10_PRCDN == 1:
        plt.plot(frac_J10[mask_J10], power_iter_UPGA_J10_PRCDN[mask_J10], ':s', markevery=10, color='green', linewidth=2, markersize=5, label='PGA (J=10, PRCDN)')
    if run_conv_PGA_J20 ==1:
        plt.plot(frac_J20[mask_J20], power_iter_conv_PGA_J20[mask_J20], ':s', markevery=10, color='green', linewidth=2, markersize=5, label='PGA (J=20)')
    if run_UPGA_J20 == 1:
        plt.plot(frac_J20[mask_J20], power_iter_UPGA_J20[mask_J20], ':s', markevery=10, color='black', linewidth=2, markersize=5, label=label_UPGA_J20)
    if run_UPGA_J10_decay == 1:
        plt.plot(frac_J10_decay[mask_J10_decay], power_iter_UPGA_J10_decay[mask_J10_decay], ':d', markevery=10, color='purple', linewidth=2, markersize=5, label=label_UPGA_J10_decay)
    if run_UPGA_J20_decay == 1:
        mask_J20_decay = frac_J20_decay_var < n_plot_outer
        plt.plot(frac_J20_decay_var[mask_J20_decay], power_iter_UPGA_J20_decay[mask_J20_decay], ':p', markevery=10, color='brown', linewidth=2, markersize=5, label=label_UPGA_J20_decay)
    if run_UPGA_J_GradReuse == 1:
        mask_J10_GradReuse = frac_J_GradReuse < n_plot_outer
        plt.plot(frac_J_GradReuse[mask_J10_GradReuse], power_iter_UPGA_J_GradReuse[mask_J10_GradReuse], ':^', markevery=10, color='teal', linewidth=2, markersize=5, label=label_UPGA_J_GradReuse)
    # plot the maximum available power (Pt)
    plt.plot(frac_J10[mask_J10], snr * np.ones_like(frac_J10[mask_J10]), '--', color='red', linewidth=2, label='Maximum Power (Pt)')
    # Mark outer-iteration boundaries with vertical grid lines
    for ii in range(1, n_plot_outer):
        plt.axvline(x=ii, color='grey', linestyle='--', linewidth=0.6, alpha=0.5)
    plt.xlabel(r'Outer iteration $I$ (inner steps shown as fractions)', fontsize="13")
    plt.ylabel(r'F x W', fontsize="13")
    plt.title(f"F x W — first {n_plot_outer} outer iterations (incl. inner)", fontsize="13")
    plt.grid(axis='y')
    plt.legend(loc='best', fontsize="12", labelspacing=0.15)
    plt.tight_layout()
    plt.savefig(directory_result + 'power_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'power_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.eps')


 