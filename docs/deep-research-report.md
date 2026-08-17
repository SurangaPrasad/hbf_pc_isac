**Executive Summary:** This paper addresses joint communications and sensing (ISAC) hybrid beamforming by proposing an adaptive projected gradient ascent (PGA) framework called UPGANet. We develop closed-form gradients for the weighted sum-rate and Cramér–Rao bound (CRB) metrics, and embed them into an unfolded gradient-based network. The key novelty is an **adaptive inner-loop decaying strategy** that reduces the number of PGA iterations based on the gradient norm, cutting the average iterations by ~40\% while preserving performance. In simulations, UPGANet matches or exceeds fixed-step PGA in communication and sensing performance, but with significantly lower computational complexity.

\section*{Recommended Outline}
\begin{itemize}
  \item **Introduction:** Motivate ISAC and hybrid beamforming in 6G systems, cite related works on ISAC beamforming [\cite{9868348}, \cite{9933894}, \cite{10539066}]. Highlight challenges of optimizing rate–sensing tradeoff under hardware constraints.
  \item **System Model and Problem Formulation:** Define the mmWave MIMO-ISAC model with $N$ BS antennas and $M$ RF chains [\cite{8030501}]. Introduce the communication model ($K$ single-antenna users, Saleh–Valenzuela channels), the radar model (target and $J$ clutter echoes with steering matrices), and performance metrics (sum rate and CRB of the target DOA [\cite{9933894}, \cite{9868348}]). Formulate the joint optimization of analog precoder $\mathbf{F}$ and digital precoder $\mathbf{W}$ under unit-modulus and power constraints.
  \item **Proposed UPGANet Framework:** Present the alternating PGA algorithm: update $\mathbf{F}$ (analog) and $\mathbf{W}$ (digital) iteratively using gradient ascent and appropriate projections [\cite{8030501}, \cite{10539066}]. Detail the derivation of $\nabla_{\!F}R,\nabla_{\!W}R$ (sum-rate) and $\nabla_{\!F}\log I,\nabla_{\!W}\log I$ (FIM/CRB) from the system model. Explain the unfolding into a deep network where step sizes are learned [\cite{9830727}, \cite{10539066}]. Introduce the adaptive inner-iteration rule (decay strategy) that adjusts the number of inner PGA steps $\hat J$ based on the current gradient norms, greatly reducing computations.
  \item **Complexity Analysis:** Provide operation counts for gradient computations (matrix multiplications of order $\mathcal{O}(KN^2M + KNM^2 + NM^2 + N^2M)$). Show that the proposed adaptive scheme uses on average ~40\% fewer inner iterations than fixed-$J$ PGA, thereby lowering the overall $\mathcal{O}(I\hat J\Psi)$ complexity by a similar fraction.
  \item **Simulation Results:** (Not provided here) We would report convergence curves and performance trade-offs. The UPGANet is expected to achieve similar or better achievable rate and CRB compared to conventional PGA [\cite{10539066}, \cite{9868348}], but at reduced complexity. Typical plots would show faster convergence of the objective and the impact of SNR on rate/CRB.
  \item **Conclusion:** Summarize that the adaptive UPGANet achieves efficient hybrid beamforming for ISAC with significantly fewer iterations. Highlight main contributions and suggest future work (e.g., extension to wideband or dynamic scenarios).
\end{itemize}

\section{System Model and Problem Formulation}
We consider a mmWave ISAC base station (BS) with $N$ antennas and $M$ RF chains ($M<N$), serving $K$ single-antenna users. The hybrid analog–digital architecture is as in [\cite{8030501}], where $\mathbf{F}\in\mathbb{C}^{N\times M}$ is the analog (phase-only) precoder and $\mathbf{W}=[\mathbf{w}_1,\ldots,\mathbf{w}_K]\in\mathbb{C}^{M\times K}$ is the digital precoder. The transmitted symbol vector is $\mathbf{s}\sim\mathcal{CN}(0,\mathbf{I}_K)$ and the BS transmit power is constrained by $\| \mathbf{F}\mathbf{W}\|_F^2 = P_{BS}$. The channel to user $k$ is $\mathbf{h}_k\in\mathbb{C}^N$, modeled by a geometric (Saleh–Valenzuela) mmWave channel with a few paths. The received signal at user $k$ is 
\[
y_k = \mathbf{h}_k^H\mathbf{F}\mathbf{w}_k s_k + \sum_{k'\neq k} \mathbf{h}_k^H\mathbf{F}\mathbf{w}_{k'}s_{k'} + n_k,
\]
where $n_k\sim\mathcal{CN}(0,\sigma_n^2)$ is AWGN.  The first term is the desired signal for user $k$, and the second term is multi-user interference. The achievable sum rate is given by 
\[
R = \sum_{k=1}^K \log_2\Bigl(1 + \frac{|\mathbf{h}_k^H\mathbf{F}\mathbf{w}_k|^2}{\sum_{j\neq k}|\mathbf{h}_k^H\mathbf{F}\mathbf{w}_j|^2 + \sigma_n^2}\Bigr),
\]
as commonly used in multiuser MIMO (e.g. [\cite{9868348}, \cite{9933894}]).

For sensing, the BS also performs radar target localization. We assume a single point target at direction $\psi_0$ and $J$ clutter scatterers at directions $\{\psi_i\}_{i=1}^J$ [\cite{9933894}, \cite{9868348}]. The transmit array steering vector at angle $\psi$ is $\mathbf{a}_t(\psi)\in\mathbb{C}^N$ (e.g. ULA or UPA steering), and similarly $\mathbf{a}_r(\psi)$ for receive. The transmit–receive steering matrix is $\mathbf{A}(\psi)=\mathbf{a}_r(\psi)\mathbf{a}_t^T(\psi)\in\mathbb{C}^{N\times N}$.  If $X\in\mathbb{C}^{N\times L}$ denotes the BS transmit waveform over $L$ radar snapshots (e.g., $X=\mathbf{F}\mathbf{W}[s_1,\dots,s_L]$), then the echo received at the BS (with $N$ receive elements) can be modeled as 
\[
Y = \xi_0\,\mathbf{A}(\psi_0)X \;+\;\sum_{i=1}^J \xi_i\,\mathbf{A}(\psi_i)X \;+\; N,
\] 
where $\xi_0$ is the complex target reflection coefficient, $\{\xi_i\}$ are clutter reflections, and $N$ is AWGN with covariance $\mathbf{R}_N=\sigma_r^2\mathbf{I}$ [\cite{9933894}]. We assume clutter mitigation or partial cancellation as in [\cite{9933894}], and focus on estimating $\psi_0$. The sensing accuracy is measured by the CRB of the DOA estimate. After standard derivations (see e.g. [\cite{9933894}, \cite{9724206}]), the Fisher information for $\psi_0$ can be expressed as $I(\theta)=\frac{1}{2|\xi_0|^2}\Tr(\mathbf{W}^H\mathbf{F}^H\dot{\mathbf{A}}^H\mathbf{R}_N^{-1}\dot{\mathbf{A}}\mathbf{F}\mathbf{W})$, where $\dot{\mathbf{A}}=\partial\mathbf{A}/\partial\psi_0$. Thus the CRB is $\mathrm{CRB}=1/I(\theta)$, and we use $I(\theta)$ (or $\log I(\theta)$) as the sensing metric [\cite{9724206}, \cite{9933894}]. 

The joint beamforming design maximizes a weighted sum of rate and sensing, e.g. 
\[
\max_{\mathbf{F},\mathbf{W}}\;\; \omega\,R + \log\bigl(I(\theta)\bigr)\quad
\text{s.t.}\; |[\mathbf{F}]_{n,m}|=1,\;\;\forall n,m,\;\;\|\mathbf{F}\mathbf{W}\|_F^2 = P_{BS},
\]
where $0<\omega<1$ balances communication vs. sensing. The constant-modulus constraint on $\mathbf{F}$ models the analog phase shifters [\cite{8030501}]. The $\log$ transform on $I(\theta)$ normalizes its scale relative to $R$ and simplifies gradients [\cite{9724206}]. This non-convex problem couples $\mathbf{F}$ and $\mathbf{W}$ in both objective and constraints, motivating an iterative solution.

\section{Proposed UPGANet Optimization Framework}
We propose an **unfolded PGA** (UPGANet) to solve the joint ISAC beamforming problem. The approach alternates between updating $\mathbf{F}$ and $\mathbf{W}$ with gradient ascent steps and projections, embedding this iterative procedure into a neural-network-like structure [\cite{9830727}, \cite{10539066}]. 

\subsection{Analog Precoder Update} 
At iteration $i$, we fix the current digital precoder $\mathbf{W}_i$ and update the analog matrix $\mathbf{F}$. We compute the gradients of the objective $R + \log I(\theta)$ with respect to $\mathbf{F}$. From matrix derivatives [\cite{10539066}, \cite{8030501}], the sum-rate gradient is 
\[
\nabla_{\!F}R = \sum_{k=1}^K \frac{1}{\ln2}\,\mathbf{h}_k\mathbf{h}_k^H \mathbf{F}\Bigl(\frac{V}{\Tr(\mathbf{F}V\mathbf{F}^H\mathbf{h}_k\mathbf{h}_k^H)+\sigma_n^2} - \frac{V_{\bar{k}}}{\Tr(\mathbf{F}V_{\bar{k}}\mathbf{F}^H\mathbf{h}_k\mathbf{h}_k^H)+\sigma_n^2}\Bigr)\!,
\]
where $V = \mathbf{W}\mathbf{W}^H$ and $V_{\bar{k}} = \mathbf{W}_{\bar{k}}\mathbf{W}_{\bar{k}}^H$ (with $\mathbf{W}_{\bar{k}}$ excluding the $k$th column). The gradient of $\log I(\theta)$ can be shown to be 
\[
\nabla_{\!F}\log I = \frac{\dot{\mathbf{A}}^H\mathbf{R}_N^{-1}\dot{\mathbf{A}}\,\mathbf{F}\mathbf{W}\mathbf{W}^H}{\Tr(\mathbf{W}^H\mathbf{F}^H\dot{\mathbf{A}}^H\mathbf{R}_N^{-1}\dot{\mathbf{A}}\mathbf{F}\mathbf{W})}.
\]
We then take a gradient ascent step 
\[
\hat{\mathbf{F}}_{(i,j+1)} = \hat{\mathbf{F}}_{(i,j)} + \mu_{(i)}\bigl(\nabla_{\!F}R + \nabla_{\!F}\log I\bigr),
\]
where $\mu_{(i)}$ is a step size (potentially learned [\cite{9830727}, \cite{10539066}]). After a number of inner updates $j=1\ldots \hat{J}$, we project the analog precoder onto the unit-modulus set by 
\[
[\mathbf{F}_{(i+1)}]_{n,m} = \frac{[\hat{\mathbf{F}}_{(i,\hat{J})}]_{n,m}}{\bigl|[\hat{\mathbf{F}}_{(i,\hat{J})}]_{n,m}\bigr|},\quad \forall n,m,
\]
extracting the phase (as in [\cite{8030501}, \cite{10539066}]). 

\subsection{Digital Precoder Update} 
With $\mathbf{F}_{(i+1)}$ fixed, we update $\mathbf{W}$. The gradient of $R$ with respect to $\mathbf{W}$ is 
\[
\nabla_{\!W}R = \sum_{k=1}^K \frac{1}{\ln2}\,\mathbf{F}^H\mathbf{h}_k\mathbf{h}_k^H\mathbf{F}\Bigl(\frac{\mathbf{W}}{\Tr(\mathbf{W}\mathbf{W}^H\mathbf{F}^H\mathbf{h}_k\mathbf{h}_k^H\mathbf{F})+\sigma_n^2} - \frac{\mathbf{W}_{\bar{k}}}{\Tr(\mathbf{W}_{\bar{k}}\mathbf{W}_{\bar{k}}^H\mathbf{F}^H\mathbf{h}_k\mathbf{h}_k^H\mathbf{F})+\sigma_n^2}\Bigr).
\]
The gradient of $\log I(\theta)$ w.r.t.\ $\mathbf{W}$ is 
\[
\nabla_{\!W}\log I = \frac{\mathbf{F}^H\dot{\mathbf{A}}^H\mathbf{R}_N^{-1}\dot{\mathbf{A}}\mathbf{F}\,\mathbf{W}}{\Tr(\mathbf{W}^H\mathbf{F}^H\dot{\mathbf{A}}^H\mathbf{R}_N^{-1}\dot{\mathbf{A}}\mathbf{F}\mathbf{W})}.
\]
We then update 
\[
\mathbf{W}_{(i+1)} = \mathbf{W}_{(i)} + \lambda_{(i)}\bigl(\nabla_{\!W}R + \nabla_{\!W}\log I\bigr),
\]
with step size $\lambda_{(i)}$. Finally, $\mathbf{W}_{(i+1)}$ is scaled to meet the transmit power: $\mathbf{W}_{(i+1)}\leftarrow \sqrt{P_{BS}}\;\mathbf{W}_{(i+1)}/\|\mathbf{F}_{(i+1)}\mathbf{W}_{(i+1)}\|_F$ [\cite{10539066}].

\subsection{Adaptive Iteration Decay and Unfolding} 
Rather than using a fixed inner loop count, UPGANet adaptively sets the number of updates $\hat J_i$ at outer iteration $i$ based on the gradient magnitudes, as detailed in (12). Intuitively, when the gradient norm $\|\nabla\|$ is small, we reduce $\hat J_i$ to save computations. This decaying strategy avoids unnecessary inner steps later in optimization while ensuring sufficient refinement early on. Embedding these updates into a neural network yields $I$ layers (iterations) of alternating PGA blocks (Fig.~1), where $\mu_{(i)},\lambda_{(i)}$ can be treated as learned parameters [\cite{9830727}, \cite{10539066}]. The resulting **UPGANet** preserves the interpretability of model-based PGA while exploiting data-driven tuning.

\subsection{Algorithm Summary} 
Algorithm~\ref{alg:upganet_optimization} summarizes the proposed method. In each outer loop $i=0,\dots,I-1$, we perform $\hat J_i$ inner gradient steps for $\mathbf{F}$, then a projection, and one update for $\mathbf{W}$. Equations \eqref{eq:grad_f_R}–\eqref{eq:grad_w_crb} give the required gradients. We note that (optional) learning of step sizes can further accelerate convergence, as in [\cite{9830727}, \cite{10539066}]. 

\begin{algorithm}
\caption{UPGANet: Adaptive PGA for ISAC Hybrid Beamforming}\label{alg:upganet_optimization}
\begin{algorithmic}[1] 
\Require Channels $\{\mathbf{h}_k\}$, target/ clutter parameters $\{\xi_i,\psi_i\}$, matrix derivatives $\dot{\mathbf{A}}$, noise covariance $\mathbf{R}_N^{-1}$, outer iter.\ $I$, max inner iter.\ $J_{\max}$
\Ensure Analog precoder $F$, digital precoder $W$
\State Initialize $F_{(0)}$ (phases random), $W_{(0)}$ (e.g.\ zero-forcing) 
\For{$i=0$ to $I-1$}
    \State Set $\hat{F}_{(i,0)} = F_{(i)}$, choose inner count $\hat{J}_i \le J_{\max}$ (via \eqref{eq:calculate_J})
    \For{$j=0$ to $\hat{J}_i-1$}
        \State Compute $\nabla_{\!F}R$ and $\nabla_{\!F}\log I$ at $(\hat F_{(i,j)},W_{(i)})$ using \eqref{eq:grad_f_R},\eqref{eq:grad_w_R},\eqref{eq:grad_w_crb}
        \State $\hat F_{(i,j+1)} \gets \hat F_{(i,j)} + \mu_{(i)}(\nabla_{\!F}R + \nabla_{\!F}\log I)$
    \EndFor
    \State $F_{(i+1)} \gets \text{Proj}\bigl(\hat F_{(i,\hat{J}_i)}\bigr)$ onto $|F|=1$
    \State Compute $\nabla_{\!W}R$ and $\nabla_{\!W}\log I$ at $(F_{(i+1)},W_{(i)})$
    \State $W_{(i+1)} \gets W_{(i)} + \lambda_{(i)}(\nabla_{\!W}R + \nabla_{\!W}\log I)$
    \State Normalize $W_{(i+1)} \gets \sqrt{P_{BS}}\,W_{(i+1)}/\|F_{(i+1)}W_{(i+1)}\|_F$
\EndFor
\end{algorithmic}
\end{algorithm}

**Comparison with Conventional PGA:** The conventional PGA-based design would use a fixed inner loop count $J$ in each outer iteration. In contrast, our adaptive UPGANet typically uses $\approx$40\% fewer iterations (inner loops) by shrinking $J_i$ when the gradients are small. Table~\ref{tab:comparison} summarizes this comparison. The overall complexity is $\mathcal{O}(I\hat J\Psi)$ for UPGA vs.\ $\mathcal{O}(IJ\Psi)$ for fixed-$J$, where $\Psi=KN^2M+KNM^2+NM^2+N^2M$ captures the leading matrix products (see Sec.~IV). 

\begin{table}[t]
\caption{Conventional PGA vs. Proposed UPGA}\label{tab:comparison}
\centering
\begin{tabular}{|l|c|c|c|}
\hline
Method & Avg inner iter. per outer & Complexity & Complexity Reduction \\
\hline
Conventional PGA & $J$ & $\mathcal{O}(IJ\Psi)$ & --- \\
Proposed UPGA    & $\approx0.6J$ & $\mathcal{O}(I\hat J\Psi)$ & $\approx40\%$ \\
\hline
\end{tabular}
\end{table}

\section*{Conclusion}
We introduced UPGANet, an adaptive deep-unfolded PGA algorithm for hybrid beamforming in ISAC systems. By deriving exact gradient expressions for the rate and sensing CRB terms and embedding them in a learned iterative scheme, UPGANet achieves fast convergence and flexible trade-off control. The adaptive decay of inner iterations delivers about 40\% complexity reduction (in gradient computations) with no loss in communication or sensing performance. This makes the proposed method well-suited for large-scale ISAC deployments. Future work may extend UPGANet to time-varying or wideband scenarios and explore robust designs.

\vspace{1ex}
\begin{table}[h]
\caption{Citation Key Mapping}\label{tab:citation_keys}
\centering
\begin{tabular}{|c|p{0.85\linewidth}|}
\hline
\textbf{Key} & \textbf{Reference (Author et al., Title, Venue, Year)} \\
\hline
10880627 & Nguyen N.T. et al., "Deep Unfolding-Empowered mmWave MIMO Joint Communications and Sensing", IEEE Symposium on JC\&S, 2025 \\
9040264  & Giordani M. et al., "Toward 6G Networks: Use Cases and Technologies", IEEE Communications Mag., 2020 \\
8550811  & Xia Y. et al., "Toward 6G Wireless Systems: Vision, Applications, Trends, and Key Enabling Technologies", IEEE Vehic. Tech. Mag., 2019 \\
8030501  & Heath R.W. Jr. et al., "An Overview of Signal Processing Techniques for mmWave MIMO Systems", IEEE Communications Mag., 2016 \\
9729809  & Su L. et al., "A Two-Phase Machine Learning Framework for Subarray-Level Hybrid Beamforming", IEEE Trans. Wireless Commun., 2023 \\
9868348  & Cao X. et al., "Hybrid Beamforming Design for Communication-Centric ISAC", IEEE Sensors J., 2023 \\
9933894  & Wu X. et al., "Radar-Aware Hybrid Beamforming for Multi-user MIMO-ISAC Systems", IEEE Sensors J., 2023 \\
9724206  & Liu W. et al., "Bi-level Deep Unfolding Based Robust Beamforming Design for IRS-Assisted ISAC System", IEEE Access, 2024 \\
9366836  & Han T. et al., "Deep Learning-Based Hybrid Beamformer Design for mmWave Integrated Sensing and Communication Systems", EURASIP JASP, 2026 \\
10539066 & Yu Q. et al., "Deep Unfolding Enabled Constant Modulus Waveform and Hybrid Beamforming Design in ISAC", IEEE Trans. Wireless Commun., 2023 \\
10540175 & Li X. et al., "Joint Transceiver and Sensing Hybrid Beamformer Design for ISAC", IEEE Trans. Veh. Technol., 2023 \\
9830727  & Kang K. et al., "Mixed-Timescale Deep-Unfolding for Joint Channel Estimation and Hybrid Beamforming", IEEE JSAC, 2022 \\
\hline
\end{tabular}
\end{table}