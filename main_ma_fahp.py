"""
Compare MA-FAHP adaptive RF-chain/antenna connection selection against the
full-connected structure, for SNR in `snr_dB_list` (0-12 dB).

For every SNR point:
  1. `ma_fahp()` searches for the best connection-state matrix D (Algorithm 2),
     using the trained `PGA_Unfold_JX_partial` (UPGA-PC) model as the CHP
     subproblem solver (see MA_FAHP.py).
  2. The objective (Eq. 10/11: OMEGA * sum-rate + mean-CRB-term - mean-power)
     is (re-)evaluated with the full outer-iteration budget for both the
     MA-FAHP-selected D and the fully-connected D (all antennas wired to all
     RF chains), so the two curves are directly comparable.
"""
import scipy.io
from MA_FAHP import *
from utility import safe_legend

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
load_saved_plot_data = 0

# Search-phase settings (cheap, only used to rank/compare candidate D's).
SEARCH_N_ITER_OUTER = 10
MAX_WHILE_LOOPS = 5
SEARCH_BATCH_SIZE = 8   # small channel subset used while searching, for speed


def get_ma_fahp_cache_file_name():
    return directory_result + 'ma_fahp_obj_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.mat'


def full_connected_matrix():
    """D with every antenna connected to every RF chain (Full-Connected structure)."""
    return np.ones((Nt, Nrf), dtype=int)


if run_program == 1:
    import time
    
    # Load test channel data
    _, H_test0 = get_data_tensor(data_source)
    H_test = H_test0[:, :test_size, :, :]
    H_search = H_test0[:, :min(SEARCH_BATCH_SIZE, test_size), :, :]

    # Load the trained UPGA-PC model whose weights will be reused for every candidate D
    model = load_pga_partial_model(step_size_UPGA_J5, model_file_name_UPGA_partial_J5)

    D_full = full_connected_matrix()

    obj_ma_fahp = np.zeros(len(snr_dB_list))
    obj_full_connected = np.zeros(len(snr_dB_list))
    active_links_ma_fahp = np.zeros(len(snr_dB_list))

    sweep_start_time = time.time()
    
    for ss, snr_dB_val in enumerate(snr_dB_list):
        snr_start_time = time.time()
        Pt = 10 ** (snr_dB_val / 10)
        print(f'\n>>> SNR sweep progress: {ss+1}/{len(snr_dB_list)}')
        print(f'---------------------- SNR = {snr_dB_val} dB ----------------------')

        params = Params(H_search, Pt, model,
                         n_iter_outer_search=SEARCH_N_ITER_OUTER,
                         n_iter_outer_eval=n_iter_outer,
                         H_eval=H_test)

        D_best = ma_fahp(params, max_while_loops=MAX_WHILE_LOOPS, verbose=True)
        active_links_ma_fahp[ss] = D_best.sum()

        # Final, fair comparison: same (full) channel batch and outer-iteration
        # budget for both connection matrices.
        print(f"  Evaluating MA-FAHP configuration on full test batch...")
        obj_ma_fahp[ss], *_ = evaluate_configuration(D_best, params, H_test, n_iter_outer)
        print(f"  Evaluating full-connected configuration on full test batch...")
        obj_full_connected[ss], *_ = evaluate_configuration(D_full, params, H_test, n_iter_outer)

        snr_elapsed = time.time() - snr_start_time
        snr_completed = ss + 1
        snr_remaining = len(snr_dB_list) - snr_completed
        avg_time_per_snr = (time.time() - sweep_start_time) / snr_completed
        est_total_remaining = avg_time_per_snr * snr_remaining
        
        print(f'  MA-FAHP objective        = {obj_ma_fahp[ss]:.4f} '
              f'(active links = {int(active_links_ma_fahp[ss])}/{Nt * Nrf})')
        print(f'  Full-connected objective = {obj_full_connected[ss]:.4f}')
        print(f'  SNR point elapsed: {snr_elapsed:.1f}s | Est. remaining: {est_total_remaining:.1f}s')

    scipy.io.savemat(get_ma_fahp_cache_file_name(), {
        'snr_dB_list': snr_dB_list,
        'obj_ma_fahp': obj_ma_fahp,
        'obj_full_connected': obj_full_connected,
        'active_links_ma_fahp': active_links_ma_fahp,
    })
    total_elapsed = time.time() - sweep_start_time
    print(f'\n=== SWEEP COMPLETE ===')
    print(f'Total elapsed time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)')
    print(f'Saved MA-FAHP objective-vs-SNR cache to', get_ma_fahp_cache_file_name())

if plot_figure == 1:
    if load_saved_plot_data == 1:
        cache = scipy.io.loadmat(get_ma_fahp_cache_file_name(), simplify_cells=True)
        snr_dB_list = np.asarray(cache['snr_dB_list'])
        obj_ma_fahp = np.asarray(cache['obj_ma_fahp'])
        obj_full_connected = np.asarray(cache['obj_full_connected'])

    plt.figure(figsize=(8, 4.2))
    plt.plot(snr_dB_list, obj_ma_fahp, '-o', color='red', linewidth=3, markersize=8,
             label='MA-FAHP (adaptive connections)')
    plt.plot(snr_dB_list, obj_full_connected, '--s', color='black', linewidth=3, markersize=8,
             label='Full-connected structure')
    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel('Objective value', fontsize=14)
    plt.grid()
    safe_legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=11, ncol=2, frameon=False)
    plt.savefig(directory_result + 'ma_fahp_obj_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png',
                bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'ma_fahp_obj_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
