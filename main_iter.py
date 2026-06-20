from PGA_models import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
save_result = 0


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


step_size_snapshots = []


def register_step_size(label, step_size_tensor):
    """Store a detached CPU copy of step sizes for post-run diagnostics."""
    if step_size_tensor is None:
        return
    if torch.is_tensor(step_size_tensor):
        step_size_snapshots.append((label, step_size_tensor.detach().cpu()))


def average_step_size_by_outer(step_size_tensor):
    """Return shape (n_outer, n_channels) averaged over inner iterations if present."""
    arr = step_size_tensor.numpy()
    if arr.ndim == 3:
        # [n_inner, n_outer, n_channels] -> [n_outer, n_channels]
        return arr.mean(axis=0)
    if arr.ndim == 2:
        # [n_outer, n_channels]
        return arr
    return None

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
        print('Running conventional PGA with J = 1...')
        model_conv_PGA_J1 = PGA_Unfold_JX(step_size_UPGA_J1)  # Reuse the same shape of step sizes as J1
        register_step_size('Conv PGA (J=1)', model_conv_PGA_J1.step_size)
        rate_conv_PGA_J1, crb_conv_PGA_J1, power_conv_PGA_J1, F_conv_PGA_J1, W_conv_PGA_J1, gradient_norm_history_conv_PGA_J1, gradient_norm_history_conv_PGA_J1_W = model_conv_PGA_J1.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J1)  # Use n_iter_inner_J1 as J=1
        rate_iter_conv_PGA_J1  = rate_conv_PGA_J1.mean(0).cpu().numpy()
        crb_iter_conv_PGA_J1   = crb_conv_PGA_J1.mean(0).cpu().numpy()
        power_iter_conv_PGA_J1 = power_conv_PGA_J1.mean(0).cpu().numpy()
    
    # ====================================================== Conv. PGA with J = 5 ====================================
    if run_conv_PGA_J5 == 1:
        print('Running conventional PGA with J = 5...')
        model_conv_PGA_J5 = PGA_Unfold_JX(step_size_UPGA_J5)  # Reuse the same shape of step sizes as J5
        register_step_size('Conv PGA (J=5)', model_conv_PGA_J5.step_size)
        rate_conv_PGA_J5, crb_conv_PGA_J5, power_conv_PGA_J5, F_conv_PGA_J5, W_conv_PGA_J5, gradient_norm_history_conv_PGA_J5, gradient_norm_history_conv_PGA_J5_W = model_conv_PGA_J5.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J5)  # Use n_iter_inner_J5 as J=5
        rate_iter_conv_PGA_J5  = rate_conv_PGA_J5.mean(0).cpu().numpy()
        crb_iter_conv_PGA_J5   = crb_conv_PGA_J5.mean(0).cpu().numpy()
        power_iter_conv_PGA_J5 = power_conv_PGA_J5.mean(0).cpu().numpy()

    # ====================================================== Conv. PGA with J = 10 ====================================
    if run_conv_PGA_J10 == 1:
        print('Running conventional PGA with J = 10...')
        model_conv_PGA_J10 = PGA_Unfold_JX(step_size_UPGA_J10)
        register_step_size('Conv PGA (J=10)', model_conv_PGA_J10.step_size)
        rate_conv_PGA_J10, crb_conv_PGA_J10, power_conv_PGA_J10, F_conv_PGA_J10, W_conv_PGA_J10, gradient_norm_history_conv_PGA_J10, gradient_norm_history_conv_PGA_J10_W = model_conv_PGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
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
        model_conv_PGA_J20 = PGA_Unfold_JX(step_size_UPGA_J20)
        register_step_size('Conv PGA (J=20)', model_conv_PGA_J20.step_size)
        rate_conv_PGA_J20, crb_conv_PGA_J20, power_conv_PGA_J20, F_conv_PGA_J20, W_conv_PGA_J20, gradient_norm_history_conv_PGA_J20, gradient_norm_history_conv_PGA_J20_W = model_conv_PGA_J20.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J20)
        rate_iter_conv_PGA_J20 = rate_conv_PGA_J20.mean(0).cpu().numpy()
        crb_iter_conv_PGA_J20  = crb_conv_PGA_J20.mean(0).cpu().numpy()
        power_iter_conv_PGA_J20 = power_conv_PGA_J20.mean(0).cpu().numpy()

    # ====================================================== Proposed Unfolded PGA light ====================================
    if run_UPGA_J1 == 1:
        print('Running unfolded PGA with J = 1...')
        # Create new model and load states
        model_UPGA_J1 = PGA_Unfold_JX(step_size_UPGA_J1)
        model_UPGA_J1.load_state_dict(torch.load(model_file_name_UPGA_J1, map_location=device))
        register_step_size('UPGA (J=1)', model_UPGA_J1.step_size)

        sum_rate_UPGA_J1, crb_UPGA_J1, power_UPGA_J1, F_UPGA_J1, W_UPGA_J1, gradient_norm_history_UPGA_J1, gradient_norm_history_UPGA_J1_W = model_UPGA_J1.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J1)
        rate_iter_UPGA_J1  = sum_rate_UPGA_J1.mean(0).cpu().numpy()
        crb_iter_UPGA_J1   = crb_UPGA_J1.mean(0).cpu().numpy()
        power_iter_UPGA_J1 = power_UPGA_J1.mean(0).cpu().numpy()

    if run_UPGA_J4 == 1:
        print('Running unfolded PGA with J = 4...')
        # Create new model and load states
        model_UPGA_J4 = PGA_Unfold_JX(step_size_UPGA_J4)
        model_UPGA_J4.load_state_dict(torch.load(directory_model + f'UPGA_J4.pth', map_location=device))
        register_step_size('UPGA (J=4)', model_UPGA_J4.step_size)
        sum_rate_UPGA_J4, crb_UPGA_J4, power_UPGA_J4, F_UPGA_J4, W_UPGA_J4, gradient_norm_history_UPGA_J4, gradient_norm_history_UPGA_J4_W = model_UPGA_J4.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J4)
        rate_iter_UPGA_J4  = sum_rate_UPGA_J4.mean(0).cpu().numpy()
        crb_iter_UPGA_J4   = crb_UPGA_J4.mean(0).cpu().numpy()
        power_iter_UPGA_J4 = power_UPGA_J4.mean(0).cpu().numpy()
    
    if run_UPGA_J5 == 1:
        print('Running unfolded PGA with J = 5...')
        # Create new model and load states
        model_UPGA_J5 = PGA_Unfold_JX(step_size_UPGA_J5)
        model_UPGA_J5.load_state_dict(torch.load(model_file_name_UPGA_J5, map_location=device))
        register_step_size('UPGA (J=5)', model_UPGA_J5.step_size)

        sum_rate_UPGA_J5, crb_UPGA_J5, power_UPGA_J5, F_UPGA_J5, W_UPGA_J5, gradient_norm_history_UPGA_J5, gradient_norm_history_UPGA_J5_W = model_UPGA_J5.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J5)
        rate_iter_UPGA_J5  = sum_rate_UPGA_J5.mean(0).cpu().numpy()
        crb_iter_UPGA_J5   = crb_UPGA_J5.mean(0).cpu().numpy()
        power_iter_UPGA_J5 = power_UPGA_J5.mean(0).cpu().numpy()
    
    if run_UPGA_J6 == 1:
        print('Running unfolded PGA with J = 6...')
        # Create new model and load states
        model_UPGA_J6 = PGA_Unfold_JX(step_size_UPGA_J6)
        model_UPGA_J6.load_state_dict(torch.load(directory_model + f'UPGA_J6.pth', map_location=device))
        register_step_size('UPGA (J=6)', model_UPGA_J6.step_size)
        sum_rate_UPGA_J6, crb_UPGA_J6, power_UPGA_J6, F_UPGA_J6, W_UPGA_J6, gradient_norm_history_UPGA_J6, gradient_norm_history_UPGA_J6_W = model_UPGA_J6.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                                snr,
                                                                                                n_iter_outer,
                                                                                                n_iter_inner_J6)
        rate_iter_UPGA_J6  = sum_rate_UPGA_J6.mean(0).cpu().numpy()
        crb_iter_UPGA_J6   = crb_UPGA_J6.mean(0).cpu().numpy()
        power_iter_UPGA_J6 = power_UPGA_J6.mean(0).cpu().numpy()

    # ====================================================== Proposed Unfolded PGA light ====================================
    if run_UPGA_J10 == 1:
        print('Running unfolded PGA with J = 10...')
        # Create new model and load states
        model_UPGA_J10 = PGA_Unfold_JX(step_size_UPGA_J10)
        model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10, map_location=device))
        register_step_size('UPGA (J=10)', model_UPGA_J10.step_size)

        sum_rate_UPGA_J10, crb_UPGA_J10, power_UPGA_J10, F_UPGA_J10, W_UPGA_J10, gradient_norm_history_UPGA_J10, gradient_norm_history_UPGA_J10_W = model_UPGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                            n_iter_inner_J10)
        # print(f'Shape of the sum_rate_UPGA_J10: {sum_rate_UPGA_J10.shape}')
        rate_iter_UPGA_J10  = sum_rate_UPGA_J10.mean(0).cpu().numpy()
        crb_iter_UPGA_J10   = crb_UPGA_J10.mean(0).cpu().numpy()
        power_iter_UPGA_J10 = power_UPGA_J10.mean(0).cpu().numpy()

    # ====================================================== Proposed Unfolded PGA ====================================
    if run_UPGA_J20 == 1:
        print('Running unfolded PGA with J = 20...')
        # Create new model and load states
        model_UPGA_J20 = PGA_Unfold_JX(step_size_UPGA_J20)
        model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))
        register_step_size('UPGA (J=20)', model_UPGA_J20.step_size)

        sum_rate_UPGA_J20, crb_UPGA_J20,power_UPGA_J20, F_UPGA_J20, W_UPGA_J20, gradient_norm_history_UPGA_J20, gradient_norm_history_UPGA_J20_W = model_UPGA_J20.execute_PGA(H_test, xi_0, A_dot, R_N_inv, snr,
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
    if run_UPGA_J5_decay == 1:
        print('Running unfolded PGA with decaying J (max J=5)...')
        model_UPGA_J5_decay = PGA_Unfold_JX_decay(step_size_UPGA_J5)
        model_UPGA_J5_decay.load_state_dict(torch.load(model_file_name_UPGA_J5_decay, map_location=device))
        register_step_size('UPGA (J=5, decay)', model_UPGA_J5_decay.step_size)

        sum_rate_UPGA_J5_decay, crb_UPGA_J5_decay, power_UPGA_J5_decay, F_UPGA_J5_decay, W_UPGA_J5_decay, gradient_norm_history_UPGA_J5_decay = model_UPGA_J5_decay.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                            n_iter_inner_J5)
        rate_iter_UPGA_J5_decay  = sum_rate_UPGA_J5_decay.mean(0).cpu().numpy()
        crb_iter_UPGA_J5_decay   = crb_UPGA_J5_decay.mean(0).cpu().numpy()
        power_iter_UPGA_J5_decay = power_UPGA_J5_decay.mean(0).cpu().numpy()
        inner_iter_history_UPGA_J5_decay = list(model_UPGA_J5_decay.inner_iter_history)
    
    
    if run_UPGA_J10_decay == 1:
        print('Running unfolded PGA with decaying J...')
        model_UPGA_J10_decay = PGA_Unfold_JX_decay(step_size_UPGA_J10_decay)
        model_UPGA_J10_decay.load_state_dict(torch.load(model_file_name_UPGA_J10_decay, map_location=device))
        register_step_size('UPGA (J=10, decay)', model_UPGA_J10_decay.step_size)

        sum_rate_UPGA_J10_decay, crb_UPGA_J10_decay, power_UPGA_J10_decay, F_UPGA_J10_decay, W_UPGA_J10_decay, gradient_norm_history_UPGA_J10_decay = model_UPGA_J10_decay.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                            n_iter_inner_J10)
        rate_iter_UPGA_J10_decay  = sum_rate_UPGA_J10_decay.mean(0).cpu().numpy()
        crb_iter_UPGA_J10_decay   = crb_UPGA_J10_decay.mean(0).cpu().numpy()
        power_iter_UPGA_J10_decay = power_UPGA_J10_decay.mean(0).cpu().numpy()
        inner_iter_history_UPGA_J10_decay = list(model_UPGA_J10_decay.inner_iter_history)
    
    if run_UPGA_J20_decay == 1:
        print('Running unfolded PGA with decaying J (max J=20)...')
        model_UPGA_J20_decay = PGA_Unfold_JX_decay(step_size_UPGA_J20_decay)
        model_UPGA_J20_decay.load_state_dict(torch.load(model_file_name_UPGA_J20_decay, map_location=device))
        register_step_size('UPGA (J=20, decay)', model_UPGA_J20_decay.step_size)

        sum_rate_UPGA_J20_decay, crb_UPGA_J20_decay, power_UPGA_J20_decay, F_UPGA_J20_decay, W_UPGA_J20_decay, gradient_norm_history_UPGA_J20_decay = model_UPGA_J20_decay.execute_PGA(H_test, xi_0, A_dot, R_N_inv,
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
        register_step_size('UPGA (J=10, GradReuse)', model_UPGA_J_GradReuse.step_size)

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
        beam_conv_PGA = get_beampattern(F_conv_PGA_J1, W_conv_PGA_J1, at, snr)
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

# Save decay inner-iteration counts in a compact MATLAB file for external plotting.
if run_program == 1:
    decay_inner_iter_data = {}
    if run_UPGA_J5_decay == 1 and 'inner_iter_history_UPGA_J5_decay' in locals():
        decay_inner_iter_data['outer_iter_J5_decay'] = np.arange(1, len(inner_iter_history_UPGA_J5_decay) + 1)
        decay_inner_iter_data['inner_iter_count_J5_decay'] = np.array(inner_iter_history_UPGA_J5_decay, dtype=np.int32)
    if run_UPGA_J10_decay == 1 and 'inner_iter_history_UPGA_J10_decay' in locals():
        decay_inner_iter_data['outer_iter_J10_decay'] = np.arange(1, len(inner_iter_history_UPGA_J10_decay) + 1)
        decay_inner_iter_data['inner_iter_count_J10_decay'] = np.array(inner_iter_history_UPGA_J10_decay, dtype=np.int32)

    if decay_inner_iter_data:
        decay_inner_iter_file_name = directory_result + 'inner_iter_count_vs_outer_' + str(Nt) + '_' + str(OMEGA) + '.mat'
        scipy.io.savemat(decay_inner_iter_file_name, decay_inner_iter_data)
        print(f'Saved inner-iteration count data to {decay_inner_iter_file_name}')

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
    frac_J1 = fractional_iters(n_iter_outer, n_iter_inner_J1)
    frac_J5 = fractional_iters(n_iter_outer, n_iter_inner_J5)
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
    if run_UPGA_J5_decay == 1:
        frac_J5_decay = fractional_iters_variable(inner_iter_history_UPGA_J5_decay)
    if run_UPGA_J10_decay == 1:
        frac_J10_decay = fractional_iters_variable(inner_iter_history_UPGA_J10_decay)
    else:
        frac_J10_decay = np.array([])
    if run_UPGA_J20_decay == 1:
        frac_J20_decay = fractional_iters_variable(inner_iter_history_UPGA_J20_decay)
    else:
        frac_J20_decay = np.array([])
    # Indices of the last inner step of each outer iteration in the flattened arrays
    # J=10: indices 10, 21, 32, ...  (block size J+1=11, last slot = J=10)
    # J=20: indices 20, 41, 62, ...  (block size J+1=21, last slot = J=20)
    outer_idx_J1 = np.arange(n_iter_inner_J1,
                             n_iter_outer * (n_iter_inner_J1 + 1),
                             n_iter_inner_J1 + 1) 
    outer_idx_J4 = np.arange(n_iter_inner_J4,
                             n_iter_outer * (n_iter_inner_J4 + 1),
                             n_iter_inner_J4 + 1)
    outer_idx_J5 = np.arange(n_iter_inner_J5,
                             n_iter_outer * (n_iter_inner_J5 + 1),
                             n_iter_inner_J5 + 1)   # length = n_iter_outer
    outer_idx_J6 = np.arange(n_iter_inner_J6,
                             n_iter_outer * (n_iter_inner_J6 + 1),
                             n_iter_inner_J6 + 1)   # length = n_iter_outer
    outer_idx_J10 = np.arange(n_iter_inner_J10,
                              n_iter_outer * (n_iter_inner_J10 + 1),
                              n_iter_inner_J10 + 1)   # length = n_iter_outer
    outer_idx_J20 = np.arange(n_iter_inner_J20,
                              n_iter_outer * (n_iter_inner_J20 + 1),
                              n_iter_inner_J20 + 1)   # length = n_iter_outer
    # outer_idx for J_decay/adaptive schedule: W-update is the LAST slot of each block
    if run_UPGA_J5_decay == 1:
        outer_idx_J5_decay = []
        _pos = 0
        for _ni in inner_iter_history_UPGA_J5_decay:
            _pos += _ni
            outer_idx_J5_decay.append(_pos)
            _pos += 1
        outer_idx_J5_decay = np.array(outer_idx_J5_decay)
        iter_outer_x_J5_decay = np.arange(1, len(outer_idx_J5_decay) + 1)
    else:
        outer_idx_J5_decay = np.array([], dtype=int)
        iter_outer_x_J5_decay = np.array([])
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
        outer_idx_J10_decay = np.array([], dtype=int)
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
        outer_idx_J20_decay = np.array([], dtype=int)
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
    plt.figure(figsize=(6.5, 3.2))
    if run_conv_PGA == 1:
        plt.plot(iter_outer_x, rate_iter_conv_PGA_J1[outer_idx_J1], '--', markevery=5, color='black', linewidth=2, markersize=6, label=Conv_PGA_J1)
    if run_conv_PGA_J5 == 1:
        plt.plot(iter_outer_x, rate_iter_conv_PGA_J5[outer_idx_J5], '--', markevery=5, color='blue', linewidth=2, markersize=6, label=Conv_PGA_J5)
    if run_conv_PGA_J10 == 1:
        plt.plot(iter_outer_x, rate_iter_conv_PGA_J10[outer_idx_J10], '-*', markevery=5, color='blue', linewidth=2, markersize=6, label=Conv_PGA_J10)
    if run_UPGA_J1 == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J1[outer_idx_J1], '-o', markevery=5, color='cyan', linewidth=2, markersize=6, label=label_UPGA_J1)
    if run_UPGA_J5 == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J5[outer_idx_J5], '--', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J5)
    if run_UPGA_J10 == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J10[outer_idx_J10], '-*', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        plt.plot(iter_outer_x, rate_iter_UPGA_J20[outer_idx_J20], '-', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J20)
    if benchmark == 1:
        plt.plot(iter_number_conv_PGA, rate_SCA, '-x', markevery=5, color='black', linewidth=2, markersize=6, label=label_SCA)
        plt.plot(iter_number_conv_PGA, rate_ZF, '-o', markevery=5, color='purple', linewidth=2, markersize=6, label=label_ZF)   
    if run_UPGA_J5_decay == 1:
        plt.plot(iter_outer_x_J5_decay, rate_iter_UPGA_J5_decay[outer_idx_J5_decay], '--', markevery=5, color='green', linewidth=2, markersize=6, label=label_UPGA_J5_decay)
    if run_UPGA_J10_decay == 1:
        plt.plot(iter_outer_x_J10_decay, rate_iter_UPGA_J10_decay[outer_idx_J10_decay], '-*', markevery=5, color='green', linewidth=2, markersize=6, label=label_UPGA_J10_decay)
    if run_UPGA_J20_decay == 1:
        plt.plot(iter_outer_x_J20_decay, rate_iter_UPGA_J20_decay[outer_idx_J20_decay], '-', markevery=5, color='green', linewidth=2, markersize=6, label=label_UPGA_J20_decay)

    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel('$R$ [bits/s/Hz]', fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12, labelspacing=0.15)
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps', bbox_inches='tight', pad_inches=0.02)

    # ==================================== inverse CRB (outer iters only) ================================================
    # plt.figure()
    # if run_conv_PGA == 1:
    #     plt.plot(iter_outer_x, crb_iter_conv_PGA_J1[outer_idx_J1], '--', markevery=5, color='blue', linewidth=2, markersize=6, label='PGA (J=1)')
    # if run_conv_PGA_J5 == 1:
    #     plt.plot(iter_outer_x, crb_iter_conv_PGA_J5[outer_idx_J5], '--', markevery=5, color='blue', linewidth=2, markersize=6, label='PGA (J=5)')
    # if run_conv_PGA_J10 == 1:
    #     plt.plot(iter_outer_x, crb_iter_conv_PGA_J10[outer_idx_J10] , '-*', markevery=5, color='blue', linewidth=2, markersize=6, label='PGA (J=10)')
    # if run_UPGA_J5 == 1:
    #     plt.plot(iter_outer_x, crb_iter_UPGA_J5[outer_idx_J5], '--', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J5)
    # if run_UPGA_J10 == 1:
    #     plt.plot(iter_outer_x, crb_iter_UPGA_J10[outer_idx_J10], '-*', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J10)
    # if run_UPGA_J20 == 1:
    #     plt.plot(iter_outer_x, crb_iter_UPGA_J20[outer_idx_J20], ':s', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J20)
    # if run_UPGA_J10_RMSProp == 1:
    #     plt.plot(iter_outer_x, crb_iter_UPGA_J10_RMSProp[outer_idx_J10], ':', markevery=5, color='green', linewidth=2, markersize=6, label='PGA (J=10, RMSProp)')
    # if run_UPGA_J10_PRCDN == 1:
    #     plt.plot(iter_outer_x, crb_iter_UPGA_J10_PRCDN[outer_idx_J10], ':*', markevery=5, color='green', linewidth=2, markersize=6, label='PGA (J=10, PRCDN)')
    # if run_UPGA_J5_decay == 1:
    #     plt.plot(iter_outer_x_J5_decay, crb_iter_UPGA_J5_decay[outer_idx_J5_decay], '--', markevery=5, color='purple', linewidth=2, markersize=6, label=label_UPGA_J5_decay)
    # if run_UPGA_J10_decay == 1:
    #     plt.plot(iter_outer_x_J10_decay, crb_iter_UPGA_J10_decay[outer_idx_J10_decay], '-*', markevery=5, color='purple', linewidth=2, markersize=6, label=label_UPGA_J10_decay)
    # if run_UPGA_J20_decay == 1:
    #     plt.plot(iter_outer_x_J20_decay, crb_iter_UPGA_J20_decay[outer_idx_J20_decay], '-', markevery=5, color='purple', linewidth=2, markersize=6, label=label_UPGA_J20_decay)
    # if run_UPGA_J_GradReuse == 1:
    #     plt.plot(iter_outer_x, 1/ np.exp(crb_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse]), ':^', markevery=5, color='teal', linewidth=2, markersize=6, label=label_UPGA_J_GradReuse)
    # plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=11)
    # plt.ylabel('Inverse CRLB', fontsize=11)
    # plt.grid()
    # safe_legend(loc='best', fontsize=9, labelspacing=0.15)
    # plt.savefig(directory_result + 'inv_crb_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    # plt.savefig(directory_result + 'inv_crb_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')


    # ==================================== CRB (outer iters only) ================================================
    plt.figure(figsize=(6.5, 3.2))
    ax = plt.gca()

    curves = []

    if run_conv_PGA == 1:
        curves.append((iter_outer_x, 1 / np.exp(crb_iter_conv_PGA_J1[outer_idx_J1]), '--', 'black', Conv_PGA_J1))

    if run_conv_PGA_J5 == 1:
        curves.append((iter_outer_x, 1 / np.exp(crb_iter_conv_PGA_J5[outer_idx_J5]), '--', 'blue', Conv_PGA_J5))

    if run_conv_PGA_J10 == 1:
        curves.append((iter_outer_x, 1 / np.exp(crb_iter_conv_PGA_J10[outer_idx_J10]), '-*', 'blue', Conv_PGA_J10))

    if run_UPGA_J5 == 1:
        curves.append((iter_outer_x, 1 / np.exp(crb_iter_UPGA_J5[outer_idx_J5]), '--', 'red', label_UPGA_J5))

    if run_UPGA_J10 == 1:
        curves.append((iter_outer_x, 1 / np.exp(crb_iter_UPGA_J10[outer_idx_J10]), '-*', 'red', label_UPGA_J10))

    if run_UPGA_J20 == 1:
        curves.append((iter_outer_x, 1 / np.exp(crb_iter_UPGA_J20[outer_idx_J20]), ':s', 'red', label_UPGA_J20))

    if run_UPGA_J5_decay == 1:
        curves.append((iter_outer_x_J5_decay, 1 / np.exp(crb_iter_UPGA_J5_decay[outer_idx_J5_decay]), '--', 'green', label_UPGA_J5_decay))

    if run_UPGA_J10_decay == 1:
        curves.append((iter_outer_x_J10_decay, 1 / np.exp(crb_iter_UPGA_J10_decay[outer_idx_J10_decay]), '-*', 'green', label_UPGA_J10_decay))

    if run_UPGA_J20_decay == 1:
        curves.append((iter_outer_x_J20_decay, 1 / np.exp(crb_iter_UPGA_J20_decay[outer_idx_J20_decay]), '-', 'green', label_UPGA_J20_decay))

    if run_UPGA_J_GradReuse == 1:
        curves.append((iter_outer_x, 1 / np.exp(crb_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse]), ':^', 'teal', label_UPGA_J_GradReuse))

    # Main plot
    for x, y, style, color, label in curves:
        ax.plot(x, y,style,markevery=5,color=color,linewidth=2,markersize=6,label=label)

    ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    ax.set_ylabel('CRLB', fontsize=14)
    ax.grid(True)

    

    # Inset zoom
    axins = inset_axes(ax,width="42%",height="38%",loc="upper right", borderpad=1.2)
    safe_legend(loc='best', fontsize=12, labelspacing=0.15)

    for x, y, style, color, label in curves:
        axins.plot(x, y,style,markevery=5,color=color,linewidth=2,markersize=5)

    # Zoom region: outer layers 80 to 100
    axins.set_xlim(80, 100)

    # Automatically choose y-limits from values inside x=[80,100]
    zoom_y_values = []
    for x, y, style, color, label in curves:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        mask = (x_arr >= 80) & (x_arr <= 100)
        if np.any(mask):
            zoom_y_values.extend(y_arr[mask])

    if len(zoom_y_values) > 0:
        y_min = np.min(zoom_y_values)
        y_max = np.max(zoom_y_values)
        y_pad = 0.12 * (y_max - y_min)
        axins.set_ylim(y_min - y_pad, y_max + y_pad)

    axins.grid(True)
    axins.tick_params(axis='both', labelsize=9)

    # Draw box and connector lines
    mark_inset(ax,axins,loc1=2,loc2=4,fc="none",ec="0.4",linewidth=1.2)

    plt.savefig(directory_result + 'crb_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png',bbox_inches='tight',pad_inches=0.02)
    plt.savefig(directory_result + 'crb_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps',bbox_inches='tight',pad_inches=0.02)

    # ===================== OBJECTIVE (outer iters only) =============================================
    plt.figure()
    fig_obj = plt.figure(5)
    if run_conv_PGA == 1:
        obj_iter_conv_PGA_J1 = OMEGA * rate_iter_conv_PGA_J1 + crb_iter_conv_PGA_J1
        plt.plot(iter_outer_x, obj_iter_conv_PGA_J1[outer_idx_J1], '--', markevery=5, color='black', linewidth=2, markersize=6, label=Conv_PGA_J1)
    if run_UPGA_J1 == 1:
        obj_iter_UPGA_J1 = OMEGA * rate_iter_UPGA_J1[outer_idx_J1] + crb_iter_UPGA_J1
        plt.plot(iter_outer_x, obj_iter_UPGA_J1, '-o', markevery=5, color='cyan', linewidth=2, markersize=6, label=label_UPGA_J1)
    if run_conv_PGA_J5 == 1:
        obj_iter_conv_PGA_J5 = OMEGA * rate_iter_conv_PGA_J5 + crb_iter_conv_PGA_J5
        plt.plot(iter_outer_x, obj_iter_conv_PGA_J5[outer_idx_J5], '--', markevery=5, color='blue', linewidth=2, markersize=6, label=Conv_PGA_J5)
    if run_conv_PGA_J10 == 1:
        obj_iter_conv_PGA_J10 = OMEGA * rate_iter_conv_PGA_J10 + crb_iter_conv_PGA_J10
        plt.plot(iter_outer_x, obj_iter_conv_PGA_J10[outer_idx_J10], '-*', markevery=5, color='blue', linewidth=2, markersize=6, label=Conv_PGA_J10)
    if run_conv_PGA_J20 == 1:
        obj_iter_conv_PGA_J20 = OMEGA * rate_iter_conv_PGA_J20 + crb_iter_conv_PGA_J20[outer_idx_J20]
        plt.plot(iter_outer_x, obj_iter_conv_PGA_J20[outer_idx_J20], '-', markevery=5, color='blue', linewidth=2, markersize=6, label=Conv_PGA_J20)
    if run_UPGA_J4 == 1:
        obj_iter_UPGA_J4 = OMEGA * rate_iter_UPGA_J4[outer_idx_J4] + crb_iter_UPGA_J4[outer_idx_J4]
        plt.plot(iter_outer_x, obj_iter_UPGA_J4, '--', markevery=5, color='orange', linewidth=2, markersize=6, label=label_UPGA_J4)
    if run_UPGA_J5 == 1:
        obj_iter_UPGA_J5 = OMEGA * rate_iter_UPGA_J5[outer_idx_J5] + crb_iter_UPGA_J5[outer_idx_J5]
        plt.plot(iter_outer_x, obj_iter_UPGA_J5, '--', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J5)
    if run_UPGA_J6 == 1:
        obj_iter_UPGA_J6 = OMEGA * rate_iter_UPGA_J6[outer_idx_J6] + crb_iter_UPGA_J6[outer_idx_J6]
        plt.plot(iter_outer_x, obj_iter_UPGA_J6, '-d', markevery=5, color='orange', linewidth=2, markersize=6, label=label_UPGA_J6)
    if run_UPGA_J10 == 1:
        obj_iter_UPGA_J10 = OMEGA * rate_iter_UPGA_J10[outer_idx_J10] + crb_iter_UPGA_J10[outer_idx_J10]
        plt.plot(iter_outer_x, obj_iter_UPGA_J10, '-*', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        obj_iter_UPGA_J20 = OMEGA * rate_iter_UPGA_J20[outer_idx_J20] + crb_iter_UPGA_J20[outer_idx_J20]
        plt.plot(iter_outer_x, obj_iter_UPGA_J20, '-', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J20)
    if run_UPGA_J5_decay == 1:
        obj_iter_UPGA_J5_decay = OMEGA * rate_iter_UPGA_J5_decay[outer_idx_J5_decay] + crb_iter_UPGA_J5_decay[outer_idx_J5_decay]
        plt.plot(iter_outer_x_J5_decay, obj_iter_UPGA_J5_decay, '--', markevery=5, color='green', linewidth=2, markersize=6, label=label_UPGA_J5_decay)
    if run_UPGA_J10_decay == 1:
        obj_iter_UPGA_J10_decay = OMEGA * rate_iter_UPGA_J10_decay[outer_idx_J10_decay] + crb_iter_UPGA_J10_decay[outer_idx_J10_decay]
        plt.plot(iter_outer_x_J10_decay, obj_iter_UPGA_J10_decay, '-*', markevery=5, color='green', linewidth=2, markersize=6, label=label_UPGA_J10_decay)
    if run_UPGA_J20_decay == 1:
        obj_iter_UPGA_J20_decay = OMEGA * rate_iter_UPGA_J20_decay[outer_idx_J20_decay] + crb_iter_UPGA_J20_decay[outer_idx_J20_decay]
        plt.plot(iter_outer_x_J20_decay, obj_iter_UPGA_J20_decay, '-', markevery=5, color='green', linewidth=2, markersize=6, label=label_UPGA_J20_decay)
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'$\omega R + \log(\text{CRLB}^{-1})$', fontsize=14)
    # plt.title("Objective function vs Iterations", fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=11, labelspacing=0.1)
    plt.savefig(directory_result + 'objective_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png',bbox_inches='tight',pad_inches=0.02)
    plt.savefig(directory_result + 'objective_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps',bbox_inches='tight',pad_inches=0.02)


    # =======================Plot Gradient Norms (inner iters)========================================
    
    fig_grad = plt.figure(6, figsize=(6.5, 3.2))
    # if run_conv_PGA == 1:
    #     grad_norms_conv_J1 = np.array(gradient_norm_history_conv_PGA_J1)
    #     plt.plot(iter_outer_x, grad_norms_conv_J1, '--', markevery=5, color='black', linewidth=2, markersize=6, label=Conv_PGA_J1)
    # if run_conv_PGA_J5 == 1:
    #     grad_norms_conv_J5 = np.array(gradient_norm_history_conv_PGA_J5)
    #     plt.plot(iter_outer_x, grad_norms_conv_J5, '--', markevery=5, color='blue', linewidth=2, markersize=6, label=Conv_PGA_J5)
    # if run_conv_PGA_J10 == 1:
    #     grad_norms_conv_J10 = np.array(gradient_norm_history_conv_PGA_J10)
    #     plt.plot(iter_outer_x, grad_norms_conv_J10, '--', markevery=5, color='red', linewidth=2, markersize=6, label=Conv_PGA_J10)
    # if run_UPGA_J1 == 1:
    #     grad_norms_UPGA_J1 = np.array(gradient_norm_history_UPGA_J1)
    #     plt.plot(iter_outer_x, grad_norms_UPGA_J1, '-d', markevery=5, color='black', linewidth=2, markersize=6, label=label_UPGA_J1)
    if run_UPGA_J5 == 1:
        grad_norms_UPGA_J5 = np.array(gradient_norm_history_UPGA_J5)
        plt.plot(iter_outer_x, grad_norms_UPGA_J5, '-d', markevery=5, color='blue', linewidth=2, markersize=6, label=label_UPGA_J5)
    if run_UPGA_J10 == 1:
        grad_norms_UPGA_J10 = np.array(gradient_norm_history_UPGA_J10)
        plt.plot(iter_outer_x, grad_norms_UPGA_J10, '-d', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J10)
    # if run_UPGA_J5_decay == 1:
    #     grad_norms_J5_decay = np.array(gradient_norm_history_UPGA_J5_decay)
    #     plt.plot(iter_outer_x_J5_decay, grad_norms_J5_decay, '--', markevery=5, color='green', linewidth=2, markersize=6, label=label_UPGA_J5_decay)
    # if run_UPGA_J10_decay == 1:
    #     grad_norms_J10_decay = np.array(gradient_norm_history_UPGA_J10_decay)
    #     plt.plot(iter_outer_x_J10_decay, grad_norms_J10_decay, '-*', markevery=5, color='purple', linewidth=2, markersize=6, label=label_UPGA_J10_decay)
    plt.xlabel(r'Outer layer index $i$', fontsize=14)
    plt.ylabel(r'Average magnitude $\vartheta^{\mathbf{F}}_{(i)}$', fontsize=14)
    # plt.title("Gradient Norm vs Iterations", fontsize=14)
    plt.grid()
    safe_legend(loc='best', fontsize=12, labelspacing=0.15)
    plt.savefig(directory_result + 'grad_norm_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'grad_norm_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps', bbox_inches='tight', pad_inches=0.02)

    # ======================Plot Gradient Norms w.r.t. W========================================

    fig_grad_W = plt.figure(7, figsize=(6.5, 3.2))
    if run_conv_PGA == 1:
        grad_norms_W_conv_J1 = np.array(gradient_norm_history_conv_PGA_J1_W)
        plt.plot(iter_outer_x, grad_norms_W_conv_J1, '--', markevery=5, color='black', linewidth=2, markersize=6, label=Conv_PGA_J1)
    if run_conv_PGA_J5 == 1:
        grad_norms_W_conv_J5 = np.array(gradient_norm_history_conv_PGA_J5_W)
        plt.plot(iter_outer_x, grad_norms_W_conv_J5, '--', markevery=5, color='blue', linewidth=2, markersize=6, label=Conv_PGA_J5)
    if run_conv_PGA_J10 == 1:
        grad_norms_W_conv_J10 = np.array(gradient_norm_history_conv_PGA_J10_W)
        plt.plot(iter_outer_x, grad_norms_W_conv_J10, '--', markevery=5, color='red', linewidth=2, markersize=6, label=Conv_PGA_J10)
    if run_UPGA_J1 == 1:
        grad_norms_W_UPGA_J1 = np.array(gradient_norm_history_UPGA_J1_W)
        plt.plot(iter_outer_x, grad_norms_W_UPGA_J1, '-d', markevery=5, color='black', linewidth=2, markersize=6, label=label_UPGA_J1)
    if run_UPGA_J5 == 1:
        grad_norms_W_UPGA_J5 = np.array(gradient_norm_history_UPGA_J5_W)
        plt.plot(iter_outer_x, grad_norms_W_UPGA_J5, '-d', markevery=5, color='blue', linewidth=2, markersize=6, label=label_UPGA_J5)
    if run_UPGA_J10 == 1:
        grad_norms_W_UPGA_J10 = np.array(gradient_norm_history_UPGA_J10_W)
        plt.plot(iter_outer_x, grad_norms_W_UPGA_J10, '-d', markevery=5, color='red', linewidth=2, markersize=6, label=label_UPGA_J10)
    
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'Avg. entry-wise magnitude of $\nabla_{\mathbf{W}}\mathcal{J}$', fontsize=14)
    # plt.title("Gradient Norm vs Iterations", fontsize=14)
    plt.grid()
    # safe_legend(loc='best', fontsize=12, labelspacing=0.15)
    plt.savefig(directory_result + 'grad_norm_W_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'grad_norm_W_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps', bbox_inches='tight', pad_inches=0.02)

    # # ===================== SAVE OUTER-ITER RESULTS TO .mat FOR MATLAB =====================
    # print('Saving outer-iteration results to .mat file...')
    # mat_data = {'iter_outer_x': iter_outer_x}
    # if run_conv_PGA == 1:
    #     mat_data['rate_conv_PGA_J1'] = rate_iter_conv_PGA_J1[outer_idx_J1]
    #     mat_data['crb_conv_PGA_J1']  = crb_iter_conv_PGA_J1[outer_idx_J1]
    #     mat_data['obj_conv_PGA_J1']  = obj_iter_conv_PGA_J1[outer_idx_J1]
    # if run_UPGA_J1 == 1:
    #     mat_data['rate_UPGA_J1'] = rate_iter_UPGA_J1[outer_idx_J1]
    #     mat_data['crb_UPGA_J1']  = crb_iter_UPGA_J1[outer_idx_J1]
    #     mat_data['obj_UPGA_J1']  = obj_iter_UPGA_J1
    # if run_conv_PGA_J5 == 1:
    #     mat_data['rate_conv_PGA_J5'] = rate_iter_conv_PGA_J5[outer_idx_J5]
    #     mat_data['crb_conv_PGA_J5']  = crb_iter_conv_PGA_J5[outer_idx_J5]
    #     mat_data['obj_conv_PGA_J5']  = obj_iter_conv_PGA_J5[outer_idx_J5]
    # if run_conv_PGA_J10 == 1:
    #     mat_data['rate_conv_PGA_J10'] = rate_iter_conv_PGA_J10[outer_idx_J10]
    #     mat_data['crb_conv_PGA_J10']  = crb_iter_conv_PGA_J10[outer_idx_J10]
    #     mat_data['obj_conv_PGA_J10']  = obj_iter_conv_PGA_J10
    # if run_conv_PGA_J20 == 1:
    #     mat_data['rate_conv_PGA_J20'] = rate_iter_conv_PGA_J20[outer_idx_J20]
    #     mat_data['crb_conv_PGA_J20']  = crb_iter_conv_PGA_J20[outer_idx_J20]
    #     mat_data['obj_conv_PGA_J20']  = obj_iter_conv_PGA_J20
    # if run_UPGA_J5 == 1:
    #     mat_data['rate_UPGA_J5'] = rate_iter_UPGA_J5[outer_idx_J5]
    #     mat_data['crb_UPGA_J5']  = crb_iter_UPGA_J5[outer_idx_J5]
    #     mat_data['obj_UPGA_J5']  = obj_iter_UPGA_J5
    # if run_UPGA_J10 == 1:
    #     mat_data['rate_UPGA_J10'] = rate_iter_UPGA_J10[outer_idx_J10]
    #     mat_data['crb_UPGA_J10']  = crb_iter_UPGA_J10[outer_idx_J10]
    #     mat_data['obj_UPGA_J10']  = obj_iter_UPGA_J10
    # if run_UPGA_J20 == 1:
    #     mat_data['rate_UPGA_J20'] = rate_iter_UPGA_J20[outer_idx_J20]
    #     mat_data['crb_UPGA_J20']  = crb_iter_UPGA_J20[outer_idx_J20]
    #     mat_data['obj_UPGA_J20']  = obj_iter_UPGA_J20
    # if run_UPGA_J10_PRCDN == 1:
    #     mat_data['rate_UPGA_J10_PRCDN'] = rate_iter_UPGA_J10_PRCDN[outer_idx_J10]
    #     mat_data['crb_UPGA_J10_PRCDN']  = crb_iter_UPGA_J10_PRCDN[outer_idx_J10]
    #     mat_data['obj_UPGA_J10_PRCDN']  = obj_iter_UPGA_J10_PRCDN
    # if run_UPGA_J10_RMSProp == 1:
    #     mat_data['rate_UPGA_J10_RMSProp'] = rate_iter_UPGA_J10_RMSProp[outer_idx_J10]
    #     mat_data['crb_UPGA_J10_RMSProp']  = crb_iter_UPGA_J10_RMSProp[outer_idx_J10]
    #     mat_data['obj_UPGA_J10_RMSProp']  = obj_iter_UPGA_J10_RMSProp
    # if run_UPGA_J5_decay == 1:
    #     mat_data['iter_outer_x_J5_decay'] = iter_outer_x_J5_decay
    #     mat_data['rate_UPGA_J5_decay']    = rate_iter_UPGA_J5_decay[outer_idx_J5_decay]
    #     mat_data['crb_UPGA_J5_decay']     = crb_iter_UPGA_J5_decay[outer_idx_J5_decay]
    #     mat_data['obj_UPGA_J5_decay']     = obj_iter_UPGA_J5_decay
    # if run_UPGA_J10_decay == 1:
    #     mat_data['iter_outer_x_J10_decay'] = iter_outer_x_J10_decay
    #     mat_data['rate_UPGA_J10_decay']    = rate_iter_UPGA_J10_decay[outer_idx_J10_decay]
    #     mat_data['crb_UPGA_J10_decay']     = crb_iter_UPGA_J10_decay[outer_idx_J10_decay]
    #     mat_data['obj_UPGA_J10_decay']     = obj_iter_UPGA_J10_decay
    # if run_UPGA_J20_decay == 1:
    #     mat_data['iter_outer_x_J20_decay'] = iter_outer_x_J20_decay
    #     mat_data['rate_UPGA_J20_decay']    = rate_iter_UPGA_J20_decay[outer_idx_J20_decay]
    #     mat_data['crb_UPGA_J20_decay']     = crb_iter_UPGA_J20_decay[outer_idx_J20_decay]
    #     mat_data['obj_UPGA_J20_decay']     = obj_iter_UPGA_J20_decay
    # if run_UPGA_J_GradReuse == 1:
    #     mat_data['rate_UPGA_J_GradReuse'] = rate_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse]
    #     mat_data['crb_UPGA_J_GradReuse']  = crb_iter_UPGA_J_GradReuse[outer_idx_J_GradReuse]
    #     mat_data['obj_UPGA_J_GradReuse']  = obj_iter_UPGA_J_GradReuse
    # mat_file_name = directory_result + 'iter_results_' + str(Nt) + '_' + str(OMEGA) + '.mat'
    # scipy.io.savemat(mat_file_name, mat_data)
    # print(f'  Saved to {mat_file_name}')

    # # ===================== OBJECTIVE INCLUDING ALL INNER ITERATIONS (first 20 outer iters) =========
    # # x-axis: fractional outer iteration so inner steps are visible between integers
    # n_plot_outer = 40   # number of outer iterations to display
    # mask_J1 = frac_J1 < n_plot_outer
    # mask_J5 = frac_J5 < n_plot_outer
    # mask_J10 = frac_J10 < n_plot_outer
    # mask_J20 = frac_J20 < n_plot_outer
    # mask_J10_decay = frac_J10_decay < n_plot_outer
    # mask_J20_decay = frac_J20_decay < n_plot_outer
    # fig_obj_inner = plt.figure(6)
    # if run_conv_PGA == 1:
    #     obj = OMEGA * rate_iter_conv_PGA_J1 + crb_iter_conv_PGA_J1
    #     plt.plot(frac_J1[outer_idx_J1], obj[outer_idx_J1], '--', markevery=10, color='blue', linewidth=2, markersize=5, label='PGA (J=1)')
    # if run_UPGA_J1 == 1:
    #     obj = OMEGA * rate_iter_UPGA_J1 + crb_iter_UPGA_J1
    #     plt.plot(frac_J1[mask_J1], obj[mask_J1], '-o', markevery=10, color='cyan', linewidth=2, markersize=5, label=label_UPGA_J1)
    # if run_conv_PGA_J10 == 1:
    #     obj = OMEGA * rate_iter_conv_PGA_J10 + crb_iter_conv_PGA_J10
    #     plt.plot(frac_J10[mask_J10], obj[mask_J10], ':*', markevery=10, color='orange', linewidth=2, markersize=5, label='PGA (J=10)')
    # if run_conv_PGA_J20 == 1:
    #     obj = OMEGA * rate_iter_conv_PGA_J20 + crb_iter_conv_PGA_J20
    #     plt.plot(frac_J20[mask_J20], obj[mask_J20], '-', markevery=10, color='black', linewidth=2, markersize=5, label='PGA (J=20)')
    # if run_UPGA_J5 == 1:
    #     obj = OMEGA * rate_iter_UPGA_J5 + crb_iter_UPGA_J5
    #     plt.plot(frac_J5[mask_J5], obj[mask_J5], ':*', markevery=10, color='orange', linewidth=2, markersize=5, label=label_UPGA_J5)
    # if run_UPGA_J10 == 1:
    #     obj = OMEGA * rate_iter_UPGA_J10 + crb_iter_UPGA_J10
    #     plt.plot(frac_J10[mask_J10], obj[mask_J10], ':*', markevery=10, color='blue', linewidth=2, markersize=5, label=label_UPGA_J10)
    # if run_UPGA_J20 == 1:
    #     obj = OMEGA * rate_iter_UPGA_J20 + crb_iter_UPGA_J20
    #     plt.plot(frac_J20[mask_J20], obj[mask_J20], '-', markevery=10, color='red', linewidth=2, markersize=5, label=label_UPGA_J20)
    # if run_UPGA_J10_RMSProp == 1:
    #     obj = OMEGA * rate_iter_UPGA_J10_RMSProp + crb_iter_UPGA_J10_RMSProp
    #     plt.plot(frac_J10[mask_J10], obj[mask_J10], ':', markevery=10, color='purple', linewidth=2, markersize=5, label='PGA (J=10, RMSProp)')
    # if run_UPGA_J10_PRCDN == 1:
    #     obj = OMEGA * rate_iter_UPGA_J10_PRCDN + crb_iter_UPGA_J10_PRCDN
    #     plt.plot(frac_J10[mask_J10], obj[mask_J10], ':s', markevery=10, color='green', linewidth=2, markersize=5, label='PGA (J=10, PRCDN)')
    # if run_UPGA_J10_decay == 1:
    #     obj = OMEGA * rate_iter_UPGA_J10_decay + crb_iter_UPGA_J10_decay
    #     plt.plot(frac_J10_decay[mask_J10_decay], obj[mask_J10_decay], ':d', markevery=10, color='purple', linewidth=2, markersize=5, label=label_UPGA_J10_decay)
    # if run_UPGA_J20_decay == 1:
    #     obj = OMEGA * rate_iter_UPGA_J20_decay + crb_iter_UPGA_J20_decay
    #     plt.plot(frac_J20_decay[mask_J20_decay], obj[mask_J20_decay], ':d', markevery=10, color='purple', linewidth=2, markersize=5, label=label_UPGA_J20_decay)
    # if run_UPGA_J_GradReuse == 1:
    #     mask_J_GradReuse = frac_J_GradReuse < n_plot_outer
    #     obj = OMEGA * rate_iter_UPGA_J_GradReuse + crb_iter_UPGA_J_GradReuse
    #     plt.plot(frac_J_GradReuse[mask_J_GradReuse], obj[mask_J_GradReuse], ':^', markevery=10, color='teal', linewidth=2, markersize=5, label='PGA (J=10, GradReuse)')
    # # Mark outer-iteration boundaries with vertical grid lines
    # for ii in range(1, n_plot_outer):
    #     plt.axvline(x=ii, color='0.75', linestyle='--', linewidth=0.6)
    # plt.xlabel(r'Outer iteration $I$ (inner steps shown as fractions)', fontsize="13")
    # plt.ylabel(r'$\omega R + 1/\mathrm{CRB}$', fontsize="13")
    # plt.title(f"Objective function — first {n_plot_outer} outer iterations (incl. inner)", fontsize="13")
    # plt.grid(axis='y')
    # safe_legend(loc='best', fontsize="12", labelspacing=0.15)
    # plt.tight_layout()
    # plt.savefig(directory_result + 'objective_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.png')
    # plt.savefig(directory_result + 'objective_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.eps')


    # # ===================== TRANSMIT POWER INCLUDING ALL INNER ITERATIONS (first 20 outer iters) =========
    # # Only J10-based models return power_fes; J20 and RMSProp do not.
    # fig_power_inner = plt.figure(7)
    # if run_conv_PGA_J10 == 1:
    #     plt.plot(frac_J10[mask_J10], power_iter_conv_PGA_J10[mask_J10], ':*', markevery=10, color='orange', linewidth=2, markersize=5, label='PGA (J=10)')
    # if run_UPGA_J10 == 1:
    #     plt.plot(frac_J10[mask_J10], power_iter_UPGA_J10[mask_J10], ':*', markevery=10, color='blue', linewidth=2, markersize=5, label=label_UPGA_J10)
    # if run_UPGA_J10_PRCDN == 1:
    #     plt.plot(frac_J10[mask_J10], power_iter_UPGA_J10_PRCDN[mask_J10], ':s', markevery=10, color='green', linewidth=2, markersize=5, label='PGA (J=10, PRCDN)')
    # if run_conv_PGA_J20 ==1:
    #     plt.plot(frac_J20[mask_J20], power_iter_conv_PGA_J20[mask_J20], ':s', markevery=10, color='green', linewidth=2, markersize=5, label='PGA (J=20)')
    # if run_UPGA_J20 == 1:
    #     plt.plot(frac_J20[mask_J20], power_iter_UPGA_J20[mask_J20], ':s', markevery=10, color='black', linewidth=2, markersize=5, label=label_UPGA_J20)
    # if run_UPGA_J10_decay == 1:
    #     plt.plot(frac_J10_decay[mask_J10_decay], power_iter_UPGA_J10_decay[mask_J10_decay], ':d', markevery=10, color='purple', linewidth=2, markersize=5, label=label_UPGA_J10_decay)
    # if run_UPGA_J20_decay == 1:
    #     plt.plot(frac_J20_decay[mask_J20_decay], power_iter_UPGA_J20_decay[mask_J20_decay], ':d', markevery=10, color='purple', linewidth=2, markersize=5, label=label_UPGA_J20_decay)
    # if run_UPGA_J_GradReuse == 1:
    #     mask_J10_GradReuse = frac_J_GradReuse < n_plot_outer
    #     plt.plot(frac_J_GradReuse[mask_J10_GradReuse], power_iter_UPGA_J_GradReuse[mask_J10_GradReuse], ':^', markevery=10, color='teal', linewidth=2, markersize=5, label=label_UPGA_J_GradReuse)
    # # plot the maximum available power (Pt)
    # plt.plot(frac_J10[mask_J10], snr * np.ones_like(frac_J10[mask_J10]), '--', color='red', linewidth=2, label='Maximum Power (Pt)')
    # # Mark outer-iteration boundaries with vertical grid lines
    # for ii in range(1, n_plot_outer):
    #     plt.axvline(x=ii, color='0.75', linestyle='--', linewidth=0.6)
    # plt.xlabel(r'Outer iteration $I$ (inner steps shown as fractions)', fontsize="13")
    # plt.ylabel(r'F x W', fontsize="13")
    # plt.title(f"F x W — first {n_plot_outer} outer iterations (incl. inner)", fontsize="13")
    # plt.grid(axis='y')
    # safe_legend(loc='best', fontsize="12", labelspacing=0.15)
    # plt.tight_layout()
    # plt.savefig(directory_result + 'power_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.png')
    # plt.savefig(directory_result + 'power_vs_all_iters_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # ===================== AVERAGE STEP SIZES ==========================================
    # Plot average over inner iterations for step_size[:,:,0] and step_size[:,:,1].
    if len(step_size_snapshots) > 0:
        # Index 0: analog/F update step size
        plt.figure(8)
        for model_label, step_tensor in step_size_snapshots:
            avg_steps = average_step_size_by_outer(step_tensor)
            if avg_steps is None or avg_steps.shape[1] <= 0:
                continue
            x_axis = np.arange(1, avg_steps.shape[0] + 1)
            plt.plot(x_axis, avg_steps[:, 0], linewidth=2, label=model_label)
        plt.xlabel(r'Outer iteration $I$', fontsize="13")
        plt.ylabel(r'Average step size $[\cdot,\cdot,0]$', fontsize="13")
        plt.title('Average Step Size Index 0 vs Outer Iteration', fontsize="13")
        plt.grid()
        safe_legend(loc='best', fontsize="11", labelspacing=0.15)
        plt.tight_layout()
        plt.savefig(directory_result + 'avg_step_size_idx0_' + str(Nt) + '_' + str(OMEGA) + '.png')
        plt.savefig(directory_result + 'avg_step_size_idx0_' + str(Nt) + '_' + str(OMEGA) + '.eps')

        # Index 1: digital/W update step size (for K=1 this is channel 1 in last dim)
        plt.figure(9)
        for model_label, step_tensor in step_size_snapshots:
            avg_steps = average_step_size_by_outer(step_tensor)
            if avg_steps is None or avg_steps.shape[1] <= 1:
                continue
            x_axis = np.arange(1, avg_steps.shape[0] + 1)
            plt.plot(x_axis, avg_steps[:, 1], linewidth=2, label=model_label)
        plt.xlabel(r'Outer iteration $I$', fontsize="13")
        plt.ylabel(r'Average step size $[\cdot,\cdot,1]$', fontsize="13")
        plt.title('Average Step Size Index 1 vs Outer Iteration', fontsize="13")
        plt.grid()
        safe_legend(loc='best', fontsize="11", labelspacing=0.15)
        plt.tight_layout()
        plt.savefig(directory_result + 'avg_step_size_idx1_' + str(Nt) + '_' + str(OMEGA) + '.png')
        plt.savefig(directory_result + 'avg_step_size_idx1_' + str(Nt) + '_' + str(OMEGA) + '.eps')


 