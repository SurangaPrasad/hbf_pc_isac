# Graph Report - .  (2026-08-10)

## Corpus Check
- 4 files · ~34,162 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 261 nodes · 476 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- PGA Gradient Math
- Training Pipeline
- System Model & CRB
- PGA/UPGA Algorithms
- ISAC Research Concepts
- SelectionNet Training
- SelectionNet Model Ref
- System Utilities
- Iteration Control
- Project Core Modules
- Opencode Graphify Plugin
- Project Agent Docs
- Sum Loss

## God Nodes (most connected - your core abstractions)
1. `Paper (LaTeX) - UPGANet adaptive PGA for ISAC hybrid beamforming` - 19 edges
2. `get_sum_rate()` - 18 edges
3. `initialize()` - 17 edges
4. `normalize()` - 17 edges
5. `Deep research report - UPGANet executive summary and outline` - 14 edges
6. `normalize_power()` - 11 edges
7. `execute_PGA - alternating F/W gradient ascent loop with projections` - 11 edges
8. `UPGANet - adaptive deep-unfolded PGA for ISAC hybrid beamforming` - 11 edges
9. `get_beam_error()` - 10 edges
10. `SelectionNet` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Paper: Joint Communications and Sensing Hybrid Beamforming Design via Deep Unfolding` --semantically_similar_to--> `Paper (LaTeX) - UPGANet adaptive PGA for ISAC hybrid beamforming`  [INFERRED] [semantically similar]
  README.md → paper.md
- `UPGANet - adaptive projected gradient ascent network` --semantically_similar_to--> `UPGANet - adaptive deep-unfolded PGA for ISAC hybrid beamforming`  [INFERRED] [semantically similar]
  deep-research-report.md → paper.md
- `get_grad_F_com - sum-rate gradient w.r.t. F` --semantically_similar_to--> `Achievable sum rate R`  [INFERRED] [semantically similar]
  description.md → paper.md
- `get_grad_F_crb - FIM/CRB gradient w.r.t. F` --semantically_similar_to--> `CRLB of DOA estimation (sensing accuracy)`  [INFERRED] [semantically similar]
  description.md → paper.md
- `get_grad_W_com - sum-rate gradient w.r.t. W` --semantically_similar_to--> `Achievable sum rate R`  [INFERRED] [semantically similar]
  description.md → paper.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **SelectionNet training pipeline modules (Files involved)** — selectionnet_py, main_selection_py, system_config_py, pga_models_py, utility_py [EXTRACTED 1.00]
- **SelectionNet backprop gradient path** — main_selection_py_run_selectionnet, pga_models_py_get_sum_loss, selectionnet_py_selectionnet, selectionnet_training_md_gumbel_softmax [EXTRACTED 1.00]
- **Differentiable discrete-assignment stack** — selectionnet_py_selectionnet, selectionnet_training_md_gumbel_softmax, selectionnet_training_md_straight_through_estimator, selectionnet_training_md_tau_annealing [INFERRED 0.85]
- **Adaptive unfolded PGA with learnable step sizes** — paper_md_upganet, deep_research_report_md_upganet, description_md_pga_unfold_j10, description_md_learnable_step_sizes [INFERRED 0.75]
- **Closed-form gradient computations driving the alternating PGA updates** — description_md_get_grad_f_com, description_md_get_grad_f_crb, description_md_get_grad_w_com, description_md_get_grad_w_crb [EXTRACTED 1.00]
- **UPGANet simulation results (convergence and SNR sweep figures)** — matlab_figure_1, matlab_figure_2, paper_md_upganet, paper_md_sum_rate, paper_md_crb [INFERRED 0.75]

## Communities (14 total, 3 thin omitted)

### Community 0 - "PGA Gradient Math"
Cohesion: 0.12
Nodes (35): clamp_complex_magnitude(), get_grad_F_com(), get_grad_F_rad(), get_grad_W_com(), get_grad_W_rad(), get_sum_loss(), PGA_Conv, PGA_Conv_comp_grad (+27 more)

### Community 1 - "Training Pipeline"
Cohesion: 0.09
Nodes (18): run_UPGA(), run_UPGA_partial(), run_UPGA_partial_decay(), clamp_complex_magnitude(), get_sum_loss(), PGA_Conv, PGA_Conv_comp_grad, PGA_Unfold_J_GradReuse (+10 more)

### Community 2 - "System Model & CRB"
Cohesion: 0.10
Nodes (30): description.md - DNN step-size training notes and code, get_crb_fe - CRB (FIM) sensing metric, get_grad_F_com - sum-rate gradient w.r.t. F, get_grad_F_crb - FIM/CRB gradient w.r.t. F, get_grad_W_com - sum-rate gradient w.r.t. W, get_grad_W_crb - FIM/CRB gradient w.r.t. W, get_sum_rate - achievable sum rate, initialize - init F and W (+22 more)

### Community 3 - "PGA/UPGA Algorithms"
Cohesion: 0.08
Nodes (8): execute_UPGA_J10_PC(), load_snr_plot_cache(), Persist SNR sweep arrays so figures can be redrawn without rerunning models., Load cached SNR sweep arrays saved by save_snr_plot_cache., save_snr_plot_cache(), get_MSE(), Add legend only when labeled artists exist to avoid Matplotlib warnings., safe_legend()

### Community 4 - "ISAC Research Concepts"
Cohesion: 0.12
Nodes (26): Deep research report - UPGANet executive summary and outline, Adaptive inner-loop decaying strategy (~40% fewer iterations via gradient norm), CRB of target DOA as sensing metric, mmWave MIMO-ISAC system model (N antennas, M RF chains, K users), Projected gradient ascent (PGA) alternating optimization, Weighted sum-rate communication metric, UPGANet - adaptive projected gradient ascent network, main_iter.py - objective value vs iterations/layers experiment (+18 more)

### Community 5 - "SelectionNet Training"
Cohesion: 0.12
Nodes (16): anneal_tau(), clip_gradients(), load_pretrained_upga(), Exponential temperature decay: high tau explores early, low tau sharpens S…, Rebuild the frozen UPGA beamformer that produces F, W. Must instantiate the…, Train SelectionNet (learnable antenna->RF-chain assignment). The sub-connected…, run_selectionnet(), SelectionNet: learnable antenna-to-RF-chain assignment for sub-connected hybrid… (+8 more)

### Community 6 - "SelectionNet Model Ref"
Cohesion: 0.13
Nodes (15): load_pretrained_upga(), run_selectionnet(), execute_PGA(), get_sum_loss(), PGA_Unfold_JX (frozen UPGA beamformer), Hand-built sub-connected mask template (PGA_models.py:406-428), SelectionNet.column_load(), SelectionNet (MLP) (+7 more)

### Community 7 - "System Utilities"
Cohesion: 0.17
Nodes (14): array_response(), extract_active_elements(), gen_channel(), get_data_tensor(), get_mat_G(), get_mat_G_SVD(), get_radar_data(), initialize_schemes() (+6 more)

### Community 8 - "Iteration Control"
Cohesion: 0.12
Nodes (14): average_step_size_by_outer(), fractional_iters_variable(), get_outer_iter_curve(), load_plot_cache(), tensor: shape [n_iter_outer, B] Returns: mean-over-batch curve of shape…, Store a detached CPU copy of step sizes for post-run diagnostics., Return shape (n_outer, n_channels) averaged over inner iterations if present., Persist the plotting arrays so plots can be regenerated without rerunning the… (+6 more)

### Community 9 - "Project Core Modules"
Cohesion: 0.29
Nodes (7): main_selection.py, matlab/comm_data.m, PGA_models.py, SelectionNet.py, Training SelectionNet (doc), system_config.py, utility.py

## Knowledge Gaps
- **29 isolated node(s):** `AGENTS.md - project knowledge graph instructions`, `system_config - system configuration for experiments`, `main_iter.py - objective value vs iterations/layers experiment`, `main_SNR.py - rate and beampattern MSE vs SNR experiment`, `main_train.py - re-training of the models` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Paper (LaTeX) - UPGANet adaptive PGA for ISAC hybrid beamforming` connect `ISAC Research Concepts` to `System Model & CRB`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `get_sum_rate()` connect `PGA Gradient Math` to `Training Pipeline`, `System Utilities`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Paper (LaTeX) - UPGANet adaptive PGA for ISAC hybrid beamforming` (e.g. with `Deep research report - UPGANet executive summary and outline` and `description.md - DNN step-size training notes and code`) actually correct?**
  _`Paper (LaTeX) - UPGANet adaptive PGA for ISAC hybrid beamforming` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `AGENTS.md - project knowledge graph instructions`, `system_config - system configuration for experiments`, `main_iter.py - objective value vs iterations/layers experiment` to the rest of the system?**
  _29 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `PGA Gradient Math` be split into smaller, more focused modules?**
  _Cohesion score 0.11756168359941944 - nodes in this community are weakly interconnected._
- **Should `Training Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.08870967741935484 - nodes in this community are weakly interconnected._
- **Should `System Model & CRB` be split into smaller, more focused modules?**
  _Cohesion score 0.09655172413793103 - nodes in this community are weakly interconnected._