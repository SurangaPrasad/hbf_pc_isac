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

SPEED OPTIMIZATION:
  - Change SPEED_PRESET from 'FAST' to 'BALANCED' or 'QUALITY' for higher accuracy
  - Change snr_dB_list to a smaller subset for quick testing:
    Example: snr_dB_list = np.array([0, 6, 12])  # Just 3 points instead of 7
"""
import scipy.io
from MA_FAHP import *
from utility import safe_legend

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
load_saved_plot_data = 0

# ============ SPEED vs QUALITY PRESETS ============
# FAST:     Complete in ~5-10 min, lower accuracy
# BALANCED: Complete in ~15-30 min, reasonable accuracy (recommended)
# QUALITY:  Complete in ~60+ min, high accuracy (original)
SPEED_PRESET = 'FAST'  # Choose: 'FAST', 'BALANCED', or 'QUALITY'

# Search-phase settings (cheap, only used to rank/compare candidate D's).
if SPEED_PRESET == 'FAST':
    SEARCH_N_ITER_OUTER = 3       # 3x faster than original (10 -> 3)
    MAX_WHILE_LOOPS = 2           # 2.5x faster than original (5 -> 2)
    SEARCH_BATCH_SIZE = 4         # Use small batch for speed
    FINAL_N_ITER_OUTER = 60       # 2x faster final eval than original (120 -> 60)
elif SPEED_PRESET == 'BALANCED':
    SEARCH_N_ITER_OUTER = 5       # Balanced
    MAX_WHILE_LOOPS = 3           # Balanced
    SEARCH_BATCH_SIZE = 6         # Reasonable batch
    FINAL_N_ITER_OUTER = 90       # Balanced final eval
else:  # QUALITY
    SEARCH_N_ITER_OUTER = 10      # Original (high quality)
    MAX_WHILE_LOOPS = 5           # Original (high quality)
    SEARCH_BATCH_SIZE = 8         # Original (high quality)
    FINAL_N_ITER_OUTER = 120      # Original (high quality)

# ============ EXPERIMENTAL: Two-phase mode ============
# If enabled: First pass with FAST settings, second pass with BALANCED/QUALITY
# This can save time by doing quick coarse search first, then refining
TWO_PHASE_MODE = False  # Set to True for faster convergence with refinement

def full_connected_matrix():
    """D with every antenna connected to every RF chain (Full-Connected structure)."""
    return np.ones((Nt, Nrf), dtype=int)


def get_ma_fahp_cache_file_name():
    """Generate the cache filename for saving/loading MA-FAHP results."""
    return directory_result + 'ma_fahp_obj_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.mat'


if run_program == 1:
    import time
    
    print(f"\n{'='*70}")
    print(f"MA-FAHP SPEED PRESET: {SPEED_PRESET.upper()}")
    print(f"{'='*70}")
    print(f"  Search iterations (per candidate):    {SEARCH_N_ITER_OUTER}")
    print(f"  Max while-loop iterations:            {MAX_WHILE_LOOPS}")
    print(f"  Search batch size:                    {SEARCH_BATCH_SIZE}")
    print(f"  Final evaluation iterations:          {FINAL_N_ITER_OUTER}")
    
    speedup_estimate = (10 * 5 * 8) / (SEARCH_N_ITER_OUTER * MAX_WHILE_LOOPS * SEARCH_BATCH_SIZE) * (120 / FINAL_N_ITER_OUTER)
    
    if SPEED_PRESET == 'FAST':
        expected_time = "~5-15 minutes"
        quality = "Lower (good for prototyping)"
    elif SPEED_PRESET == 'BALANCED':
        expected_time = "~15-30 minutes"
        quality = "Reasonable (recommended for results)"
    else:
        expected_time = "~60+ minutes"
        quality = "High (for publication-quality results)"
    
    print(f"\n  Estimated speedup vs QUALITY:         ~{speedup_estimate:.1f}x")
    print(f"  Expected total runtime:               {expected_time}")
    print(f"  Result quality:                       {quality}")
    print(f"{'='*70}\n")
    
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
    
    # Store components for plotting
    sum_rate_ma_fahp = np.zeros(len(snr_dB_list))
    sum_rate_full = np.zeros(len(snr_dB_list))
    crb_inv_log_ma_fahp = np.zeros(len(snr_dB_list))
    crb_inv_log_full = np.zeros(len(snr_dB_list))
    power_ma_fahp = np.zeros(len(snr_dB_list))
    power_full = np.zeros(len(snr_dB_list))

    sweep_start_time = time.time()
    
    for ss, snr_dB_val in enumerate(snr_dB_list):
        snr_start_time = time.time()
        Pt = 10 ** (snr_dB_val / 10)
        print(f'\n>>> SNR sweep progress: {ss+1}/{len(snr_dB_list)} | SNR = {snr_dB_val} dB')
        print(f'    (est. time remaining: ~{(len(snr_dB_list) - ss - 1) * (time.time() - sweep_start_time) / (ss + 1):.0f}s)')
        print(f'    {"="*60}')

        params = Params(H_search, Pt, model,
                         n_iter_outer_search=SEARCH_N_ITER_OUTER,
                         n_iter_outer_eval=FINAL_N_ITER_OUTER,
                         H_eval=H_test)

        D_best = ma_fahp(params, max_while_loops=MAX_WHILE_LOOPS, verbose=True)
        active_links_ma_fahp[ss] = D_best.sum()

        # Final, fair comparison: same (full) channel batch and outer-iteration
        # budget for both connection matrices.
        print(f"  Evaluating MA-FAHP configuration on full test batch...")
        obj_ma_fahp[ss], sum_rate_ma_fahp[ss], crb_mean_ma, power_ma_fahp[ss] = evaluate_configuration(D_best, params, H_test, FINAL_N_ITER_OUTER)
        crb_inv_log_ma_fahp[ss] = crb_mean_ma
        
        print(f"  Evaluating full-connected configuration on full test batch...")
        obj_full_connected[ss], sum_rate_full[ss], crb_mean_full, power_full[ss] = evaluate_configuration(D_full, params, H_test, FINAL_N_ITER_OUTER)
        crb_inv_log_full[ss] = crb_mean_full

        snr_elapsed = time.time() - snr_start_time
        snr_completed = ss + 1
        snr_remaining = len(snr_dB_list) - snr_completed
        avg_time_per_snr = (time.time() - sweep_start_time) / snr_completed
        est_total_remaining = avg_time_per_snr * snr_remaining
        
        print(f'  MA-FAHP objective        = {obj_ma_fahp[ss]:.4f} (active links = {int(active_links_ma_fahp[ss])}/{Nt * Nrf})')
        print(f'    ├─ Sum Rate           = {sum_rate_ma_fahp[ss]:.4f}')
        print(f'    ├─ log10(CRB^-1)      = {crb_inv_log_ma_fahp[ss]:.4f}')
        print(f'    └─ Power              = {power_ma_fahp[ss]:.4f}')
        print(f'  Full-connected objective = {obj_full_connected[ss]:.4f}')
        print(f'    ├─ Sum Rate           = {sum_rate_full[ss]:.4f}')
        print(f'    ├─ log10(CRB^-1)      = {crb_inv_log_full[ss]:.4f}')
        print(f'    └─ Power              = {power_full[ss]:.4f}')
        print(f'  SNR point elapsed: {snr_elapsed:.1f}s | Est. remaining: {est_total_remaining:.1f}s')

    scipy.io.savemat(get_ma_fahp_cache_file_name(), {
        'snr_dB_list': snr_dB_list,
        'obj_ma_fahp': obj_ma_fahp,
        'obj_full_connected': obj_full_connected,
        'active_links_ma_fahp': active_links_ma_fahp,
        'sum_rate_ma_fahp': sum_rate_ma_fahp,
        'sum_rate_full': sum_rate_full,
        'crb_inv_log_ma_fahp': crb_inv_log_ma_fahp,
        'crb_inv_log_full': crb_inv_log_full,
        'power_ma_fahp': power_ma_fahp,
        'power_full': power_full,
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
        sum_rate_ma_fahp = np.asarray(cache.get('sum_rate_ma_fahp', np.zeros_like(snr_dB_list)))
        sum_rate_full = np.asarray(cache.get('sum_rate_full', np.zeros_like(snr_dB_list)))
        crb_inv_log_ma_fahp = np.asarray(cache.get('crb_inv_log_ma_fahp', np.zeros_like(snr_dB_list)))
        crb_inv_log_full = np.asarray(cache.get('crb_inv_log_full', np.zeros_like(snr_dB_list)))
        power_ma_fahp = np.asarray(cache.get('power_ma_fahp', np.zeros_like(snr_dB_list)))
        power_full = np.asarray(cache.get('power_full', np.zeros_like(snr_dB_list)))

    # ========== Plot 1: Objective Function ==========
    plt.figure(figsize=(10, 5))
    plt.plot(snr_dB_list, obj_ma_fahp, '-o', color='red', linewidth=3, markersize=8,
             label='MA-FAHP (adaptive)')
    plt.plot(snr_dB_list, obj_full_connected, '--s', color='black', linewidth=3, markersize=8,
             label='Full-connected')
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Objective value', fontsize=12)
    plt.title('Objective Function vs SNR', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    safe_legend(loc='best', fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(directory_result + 'ma_fahp_objective_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png',
                bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.savefig(directory_result + 'ma_fahp_objective_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
    
    # ========== Plot 2: Sum Rate ==========
    plt.figure(figsize=(10, 5))
    plt.plot(snr_dB_list, sum_rate_ma_fahp, '-o', color='red', linewidth=3, markersize=8,
             label='MA-FAHP (adaptive)')
    plt.plot(snr_dB_list, sum_rate_full, '--s', color='black', linewidth=3, markersize=8,
             label='Full-connected')
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Sum Rate [bits/s/Hz]', fontsize=12)
    plt.title('Sum Rate vs SNR', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    safe_legend(loc='best', fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(directory_result + 'ma_fahp_sum_rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png',
                bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.savefig(directory_result + 'ma_fahp_sum_rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
    
    # ========== Plot 3: log10(CRB^-1) ==========
    plt.figure(figsize=(10, 5))
    plt.plot(snr_dB_list, crb_inv_log_ma_fahp, '-o', color='red', linewidth=3, markersize=8,
             label='MA-FAHP (adaptive)')
    plt.plot(snr_dB_list, crb_inv_log_full, '--s', color='black', linewidth=3, markersize=8,
             label='Full-connected')
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('log₁₀(CRB⁻¹)', fontsize=12)
    plt.title('Sensing Performance (log10(CRB Inverse)) vs SNR', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    safe_legend(loc='best', fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(directory_result + 'ma_fahp_crb_inv_log_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png',
                bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.savefig(directory_result + 'ma_fahp_crb_inv_log_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
    
    # ========== Plot 4: Power Consumption ==========
    plt.figure(figsize=(10, 5))
    plt.plot(snr_dB_list, power_ma_fahp, '-o', color='red', linewidth=3, markersize=8,
             label='MA-FAHP (adaptive)')
    plt.plot(snr_dB_list, power_full, '--s', color='black', linewidth=3, markersize=8,
             label='Full-connected')
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Total Power [W]', fontsize=12)
    plt.title('Power Consumption vs SNR', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    safe_legend(loc='best', fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(directory_result + 'ma_fahp_power_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png',
                bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.savefig(directory_result + 'ma_fahp_power_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
    
    # ========== Plot 5: All metrics in one figure (subplots) ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MA-FAHP Performance Comparison (Nt={Nt}, OMEGA={OMEGA})', fontsize=14, fontweight='bold')
    
    # Objective
    axes[0, 0].plot(snr_dB_list, obj_ma_fahp, '-o', color='red', linewidth=2.5, markersize=7, label='MA-FAHP')
    axes[0, 0].plot(snr_dB_list, obj_full_connected, '--s', color='black', linewidth=2.5, markersize=7, label='Full-connected')
    axes[0, 0].set_xlabel('SNR [dB]', fontsize=11)
    axes[0, 0].set_ylabel('Objective', fontsize=11)
    axes[0, 0].set_title('Objective Function', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    # Sum Rate
    axes[0, 1].plot(snr_dB_list, sum_rate_ma_fahp, '-o', color='red', linewidth=2.5, markersize=7, label='MA-FAHP')
    axes[0, 1].plot(snr_dB_list, sum_rate_full, '--s', color='black', linewidth=2.5, markersize=7, label='Full-connected')
    axes[0, 1].set_xlabel('SNR [dB]', fontsize=11)
    axes[0, 1].set_ylabel('Sum Rate [bits/s/Hz]', fontsize=11)
    axes[0, 1].set_title('Communication: Sum Rate', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    
    # log10(CRB^-1)
    axes[1, 0].plot(snr_dB_list, crb_inv_log_ma_fahp, '-o', color='red', linewidth=2.5, markersize=7, label='MA-FAHP')
    axes[1, 0].plot(snr_dB_list, crb_inv_log_full, '--s', color='black', linewidth=2.5, markersize=7, label='Full-connected')
    axes[1, 0].set_xlabel('SNR [dB]', fontsize=11)
    axes[1, 0].set_ylabel('log₁₀(CRB⁻¹)', fontsize=11)
    axes[1, 0].set_title('Sensing: log10(CRB Inverse)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    
    # Power
    axes[1, 1].plot(snr_dB_list, power_ma_fahp, '-o', color='red', linewidth=2.5, markersize=7, label='MA-FAHP')
    axes[1, 1].plot(snr_dB_list, power_full, '--s', color='black', linewidth=2.5, markersize=7, label='Full-connected')
    axes[1, 1].set_xlabel('SNR [dB]', fontsize=11)
    axes[1, 1].set_ylabel('Total Power [W]', fontsize=11)
    axes[1, 1].set_title('Power Efficiency', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(directory_result + 'ma_fahp_all_metrics_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png',
                bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.savefig(directory_result + 'ma_fahp_all_metrics_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps',
                bbox_inches='tight', pad_inches=0.02)
    
    print(f"\n✓ Plots saved to {directory_result}")
    print(f"  - ma_fahp_objective_vs_SNR_*.{{'png','eps'}}")
    print(f"  - ma_fahp_sum_rate_vs_SNR_*.{{'png','eps'}}")
    print(f"  - ma_fahp_crb_inv_log_vs_SNR_*.{{'png','eps'}}")
    print(f"  - ma_fahp_power_vs_SNR_*.{{'png','eps'}}")
    print(f"  - ma_fahp_all_metrics_vs_SNR_*.{{'png','eps'}} (combined 2x2 subplot)")
