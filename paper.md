\section{Introduction}
Integrated sensing and communication (ISAC) has emerged as a crucial technology for future sixth-generation (6G) wireless networks, aiming to realize communication and sensing capabilities within a single unified framework. The core idea of ISAC is that communication and sensing functions jointly utilize spectrum, waveforms, and signaling resources, thereby enabling more efficient use of network resources. Through this integration, ISAC provides several significant benefits, including enhanced spectral efficiency, lower hardware costs, reduced energy consumption, and improved synchronization, all of which are vital for the sustainable evolution of wireless communications. 

To effectively accommodate the high data rates and directional transmission demands of mmWave and massive MIMO ISAC systems, hybrid beamforming (HBF) has emerged as a highly promising transmission strategy. In HBF architectures, the beamforming operation is split between the digital and analog domains, allowing large antenna arrays to realize substantial beamforming gains while considerably reducing hardware complexity and power consumption relative to fully digital beamforming schemes \cite{Molisch2017HybridSurvey,Yu2016Alternating,Sohrabi2016Hybrid}. Thanks to these benefits, HBF has become a focal research area in ISAC, and a wide range of studies have examined the design and optimization of HBF transmitters for integrated sensing and communication tasks. Prior works have explored hybrid beamforming in dual-function radar-communication systems \cite{Cheng2021HybridMultiCarrier,Cheng2021JSTSP,Qi2022HybridISAC}, in partially connected HBF structures \cite{Wang2022PartiallyConnected}, and in full-duplex ISAC beamforming configurations \cite{Liyanaarachchi2021FullDuplex,Barneto2022Beamformer,Islam2022FDISAC}. In addition, hardware-efficient and THz-based HBF approaches have been investigated in \cite{Elbir2021THz,Elbir2021ModelBased,Kaushik2021HardwareEfficient}, while recent progress in artificial intelligence and deep unfolding has led to more efficient and adaptive HBF optimization frameworks for ISAC systems \cite{Nguyen2024DeepUnfoldingISAC,Nguyen2025DeepUnfoldingJCAS,Shlezinger2024AIEmpowered}.



Recently, model-based machine learning techniques have attracted significant attention for analog and digital precoder optimization in ISAC systems, as they combine the interpretability and domain knowledge of conventional optimization algorithms with the fast inference capability of deep learning methods. In particular, deep unfolding approaches have emerged as an effective framework for hybrid beamforming optimization, where iterative signal processing algorithms are transformed into trainable neural network architectures with reduced computational complexity and improved convergence behavior. Several recent works have applied model-based learning for HBF design in massive MIMO and ISAC systems. For example, deep unfolding methods for joint communication and sensing beamforming optimization were investigated in \cite{Nguyen2024DeepUnfoldingISAC,Nguyen2025DeepUnfoldingJCAS}, while AI-assisted hybrid MIMO beamforming frameworks were studied in \cite{Shlezinger2024AIEmpowered}. Related model-based and model-free learning strategies for THz joint radar-communication systems were presented in \cite{Elbir2021ModelBased}, and rapid learning-based hybrid precoding methods were developed in \cite{Agiv2022Rapidly}. In addition, end-to-end learning and autoencoder-based ISAC optimization frameworks have also been explored in \cite{MateosRamos2022E2E,Muth2023Autoencoder}, demonstrating the growing importance of machine learning-driven precoder optimization for future intelligent ISAC networks.


\section{SYSTEM MODEL AND PROBLEM FORMULATION}

In this section, we consider a MIMO Joint Communication and Sensing (JCAS) system where a base station (BS) equipped with $N$ antennas simultaneously serves $K$ single-antenna communication users and performs radar sensing for target localization. We adopt a hybrid beamforming architecture at the BS, utilizing $M$ RF chains to balance hardware cost and beamforming gain.


\subsection{Communication Signal Model}

The BS transmits a joint signal $\mathbf{x} \in \mathbb{C}^{N \times 1}$, which can be expressed as:

\begin{equation}
    x = FWs
\end{equation}

where $\mathbf{F} \in \mathbb{C}^{N \times M}$ is the analog beamformer, $\mathbf{W} = [w_1, w_2, ...w_k] \in \mathbb{C}^{M \times K}$ is the digital precoder, and $\mathbf{s} \in \mathbb{C}^{K \times 1}$ contains the data symbols for the $K$ users. The received signal at the $k$-th user is given by:

\begin{equation}
\label{eq:received_signal}
    y_k = h_k^H F w_k s_k + \sum_{k' \neq k}^K h_k^H F w_{k'} s_{k'} + n_k
\end{equation}

where $\mathbf{h}_k \in \mathbb{C}^{N \times 1}$ is the channel vector from the BS to user $k$, modeled using the Saleh-Valenzuela model, and $n_k \sim \mathcal{CN}(0, \sigma_n^2)$ is the additive white Gaussian noise (AWGN). The first term represents the desired signal, while the second term accounts for the multi-user interference (MUI).

\subsection{Radar Sensing Model}

For the sensing task, we assume the BS aims to estimate the direction of arrival (DOA) of a target located at angle $\psi_0$. In the presence of $J$ signal-dependent clutters (e.g., reflections from trees or buildings) at angles $\psi_j$, the received echo signal $\mathbf{Y}$ at the BS is modeled as:

\begin{equation}
    Y = \xi_0 A(\psi_0)X + \sum_{i=1}^J \xi_i A(\psi_i)X + N
\end{equation}

where $\xi_0$ and $\xi_i$ are the complex reflection coefficients for the target and the $i$-th interference, respectively. $\mathbf{A}(\psi) = \mathbf{a}_r(\psi)\mathbf{a}_t^T(\psi)$ represents the transmit-receive steering matrix. $\mathbf{N} \in \mathbb{C}^{\mathbf{N}_{r}\times L}$ is the AWGN

\subsection{Problem Formulation}

Our goal is to optimize the hybrid beamformers to maximize the communication sum rate while ensuring a high sensing accuracy, characterized by the Cramér-Rao lower bound (CRLB).

\begin{enumerate}
    \item Sum Rate: The achievable sum rate for the $K$ users is defined as:
    \begin{equation}
        R = \sum_{k=1}^K \log_2 \left( 1 + \frac{|\mathbf{h}_k^H \mathbf{F} \mathbf{w}_k|^2}{\sum_{k' \neq k}^K |\mathbf{h}_k^H \mathbf{F} \mathbf{w}_{k'}|^2 + \sigma_n^2} \right)
    \end{equation}

    \item Sensing Performance: To evaluate sensing, we use the CRLB of the DOA estimation for $\psi_0$. Following the clutter cancellation, the CRLB is expressed as:

    \begin{equation}
        \begin{aligned}
            \text{CRB} &= \frac{1}{2|\xi_0|^2} \left( \text{Tr}\left( \mathbf{W}^H \mathbf{F}^H \dot{\mathbf{A}}^H \mathbf{R}_N^{-1} \dot{\mathbf{A}} \mathbf{F} \mathbf{W} \right) \right)^{-1} \\
            I(\theta) &= \frac{1}{\text{CRB}} 
        \end{aligned}
    \end{equation}

    where $I(\theta)$ is the fisher information matrix (\textbf{FIM}) and $\dot{\mathbf{A}}$ is the derivative of the steering matrix with respect to $\psi_0$. $\mathbf{R}_N$ is the noise covariance matrix.
\end{enumerate}

Based on the defined metrics, our objective is to jointly design the analog beamformer $\mathbf{F}$ and the digital precoder $\mathbf{W}$ to maximize a weighted performance metric. The optimization problem is formulated as follows:

% Problem Formulation: Objective and Constraints
\begin{equation}
    \begin{aligned}
    & \max_{F, W} \quad \omega R + \log\left(I(\theta)\right) \\
    & \text{s.t.} \quad |[F]_{n,m}| = 1, \quad \forall n, m \\
    & \quad \quad \|Fw\|_F^2 = P_{BS}
    \end{aligned}
\end{equation}

where the first constraint accounts for the constant modulus requirement of the phase shifters in the analog domain, and the second constraint ensures the total transmit power at the BS is limited to $P_{BS}$. Notably, we employ the term $\log(I(\theta)$ in the objective function rather than the CRLB itself for two primary reasons. First, because the numerical value of the CRLB is typically much smaller than the sum rate, the logarithm helps bring both metrics to a comparable scale. Second, this transformation simplifies the gradient calculations during the optimization process, thereby reducing the overall computational complexity of the unfolding algorithm.

\section{Proposed Solution}

\subsection{Proposed PGA Optimization Framework}

To solve the joint optimization problem, we propose a modified version of alternating optimization (AO) framework, leveraging with projected gradient ascent (PGA). Since the optimization of the analog beamformer $\mathbf{F}$ and digital precoder $\mathbf{W}$ is non-convex and highly coupled, we optimize them iteratively. In each iteration, one variable is updated via PGA while holding the other fixed.

\subsection{Alternating Optimization Steps}

In the $(i+1)$-th iteration, the analog beamformer $\mathbf{F}$ is updated by moving in the direction of the gradient of the objective function, followed by a projection onto the constant modulus constraint:

% Analog Beamformer Update
\begin{equation}
\label{eq:calculate_f_i}
F_{(i+1)} = F_{(i)} + \mu_{(i)} \nabla_{F} \left(R + \log\left(I(\theta)\right)\right)
\end{equation}

% Constant Modulus Projection
\begin{equation}
[F_{(i+1)}]_{n,m} = \frac{[F_{(i+1)}]_{n,m}}{|[F_{(i+1)}]_{n,m}|}, \quad \forall n, m
\end{equation}

where $\mu_{(i)}$ is the step size. Similarly, we update the digital precoder $\mathbf{W}$ via gradient ascent, followed by a normalization step to satisfy the total power constraint:

% Digital Precoder Update
\begin{equation}
W_{(i+1)} = W_{(i)} + \lambda_{(i)} \nabla_{W} \left(R + \log\left(I(\theta)\right)\right)
\end{equation}

% Power Normalization
\begin{equation}
\label{eq:projection_w}
W_{(i+1)} = \frac{\sqrt{P_{BS}}W_{(i+1)}}{\|F_{(i+1)}W_{(i+1)}\|_F}
\end{equation}

where $\lambda_{(i)}$ denotes the step size for the precoder update.

\subsection{Gradient Computations}
The gradients of the sum rate and the FIM with respect to the optimization variables are derived below.

\begin{enumerate}
    \item Sum Rate Gradients:
    Following matrix calculus and based on your derivations, the gradient of the sum rate $R$ with respect to $F$ is:

    % Sum Rate Gradient w.r.t F (Split for better layout)
    \begin{figure*}[!t]
    \begin{equation}
        \label{eq:grad_f_R} % Label is safe here or right before \end{equation}
        \begin{split}
            \nabla_{F} R = \sum_{k=1}^K \epsilon \tilde{H}_k F \Bigg(\frac{V}{\text{Tr}(FVF^H\tilde{H}_k) + \sigma_n^2}
             - \frac{V_{\bar{k}}}{\text{Tr}(FV_{\bar{k}}F^H\tilde{H}_k) + \sigma_n^2} \Bigg)
        \end{split}
    \end{equation}
    \end{figure*}
    where $\epsilon = 1/\ln 2$, $\tilde{H}_k = h_k h_k^H$, $V = WW^H$, and $V_{\bar{k}} = W_{\bar{k}}W_{\bar{k}}^H$. Here, $W_{\bar{k}}$ is the precoder matrix with the $k$-th column zeroed out.

    The gradient with respect to $\mathbf{W}$ is given by:

        % Sum Rate Gradient w.r.t W (Split for better layout)
        \begin{figure*}[!t]
            \begin{equation}
            \label{eq:grad_w_R}
            \begin{split}
            \nabla_{W} R = \sum_{k=1}^K \epsilon \bar{H}_k \Bigg( & \frac{W}{\text{Tr}(WW^H\bar{H}_k) + \sigma_n^2} 
            - \frac{W_{\bar{k}}}{\text{Tr}(W_{\bar{k}}W_{\bar{k}}^H\bar{H}_k) + \sigma_n^2} \Bigg)
            \end{split}
            \end{equation}
        \end{figure*}

    where $\bar{\mathbf{H}}_k = \mathbf{F}^H \tilde{\mathbf{H}}_k \mathbf{F}$.

    
    \item FIM Gradients:

    From the implementation logic of computing gradients for $\log(I(\theta))$, let $\mathbf{M} = \dot{\mathbf{A}}^H \mathbf{R}_N^{-1} \dot{\mathbf{A}}$. The gradients derived from the trace operations yield:

        % CRB Gradient w.r.t F
        \begin{equation}
        \nabla_{F} \log\left(I(\theta)\right) = \frac{M F W W^H}{\text{Tr}(W^H F^H M F W)}
        \end{equation}
        
        % CRB Gradient w.r.t W
        \begin{equation}
        \label{eq:grad_w_crb}
        \nabla_{W} \log\left(I(\theta)\right) = \frac{F^H M FW}{\text{Tr}(W^H F^H M F W)}
        \end{equation}
\end{enumerate}


\subsection{Calculating J}

The adaptive inner iteration count $\hat{J}$ is determined using the overall gradient magnitude of the communication and sensing objectives. First, the normalized gradient norm $\|\nabla\|$ is computed from the Frobenius norms of $\nabla_{\mathbf{F}} R$ and $\nabla_{\mathbf{F}} I(\theta)$. Then, an adaptive ratio $r \in [0,1]$ is calculated to scale the maximum iteration count $J$. Finally, $\hat{J}$ is constrained between $2$ and $J$, allowing larger gradients to use more inner iterations while reducing computational complexity for smaller gradients.

    \begin{equation}
        \label{eq:calculate_J}
        \begin{aligned}
            \left\| \nabla \right\|
            &=
            \frac{
            \left\| \nabla_{F} R \right\|_{F}
            }{\left\| \nabla_{F} R \right\|} + 
            \frac{\left\| \nabla_{F} I(\theta) \right\|_{F}
            }{\left\| \nabla_{F} I(\theta) \right\|},
            \\
            r
            &=
            \frac{
            \left\| \nabla \right\|
            }{
            \left\| \nabla \right\| + \alpha + \epsilon
            },
            \\
            \hat{J}
            &=
            \max \left(
            2,\,
            \min \left(
            J,\,
            \left\lceil Jr \right\rceil
            \right)
            \right).
        \end{aligned}
    \end{equation}

\begin{algorithm}
\caption{Proposed UPGANet based optimization}\label{alg:upganet_optimization}

    
\begin{algorithmic}[1] 
    \renewcommand{\algorithmicrequire}{\textbf{Input:}}
    \renewcommand{\algorithmicensure}{\textbf{Output:}}
    \Require $H, P_{BS}, \xi, \dot{{A}}$, ${R}_N^{-1}, I$ and $J$
    \Ensure Optimized parameters $F$ and $W$
    \State \textbf{Initialization 1:}  $F_{(0)}$ and $W_{(0)}$
    \State $i = 0$ and $\hat{J} = J$
    \While{$i < I$}
        \State set $\hat{F}_{(i,0)} = F_{(i)}$
        \State $\hat{j} = 0$
        \While{$\hat{j} < \hat{J}$}
            \State Obtain the gradients $\nabla_{\mathbf{F}} R$ and $\nabla_{\mathbf{F}} I(\theta)$ at
            \Statex \hspace{\algorithmicindent}\hspace{\algorithmicindent}$(\mathbf{F}, \mathbf{W}) = (\hat{\mathbf{F}}_{(i,\hat{j})}, \mathbf{W}_i)$ using equations \eqref{eq:grad_f_R}, \eqref{eq:grad_w_R}.
            \State Obtain $F_{(i, \hat{j}+1)}$ using the equations \eqref{eq:calculate_f_i}
            \State $\hat{j} = \hat{j}+1$
        \EndWhile
        \State Obtain $\hat{J}$ using the equation \eqref{eq:calculate_J}
        \State Set $F_{(i + 1)} = \hat{F_{(i,\hat{J})}}$ and apply projection 
        \State Obtain the gradients $\nabla_W R$ and $\nabla_W I(\theta)$ at $(F, W) = $
        \Statex \hspace{\algorithmicindent} $(F_{i+1}, W_i)$ based on the equations \eqref{eq:grad_w_R} and \eqref{eq:grad_w_crb}
        \State Obtain $W_{i+1}$ and apply the projection in equation \ref{eq:projection_w}
\EndWhile
\State return $F_I, W_I$
\end{algorithmic}
\end{algorithm}

\section{Complexity Analysis}

The computational complexity of the proposed UPGANet optimization framework is mainly determined by the gradient calculations and the iterative updates of the hybrid beamformers. For the communication objective, the dominant operations arise from matrix multiplications associated with the rate gradients in \eqref{eq:grad_f_R} and \eqref{eq:grad_w_R}. Specifically, the computation of $\nabla_F R$ requires matrix products involving $\tilde{H}_k F V$ and trace operations, resulting in a complexity of approximately $\mathcal{O}(KN^2M + KNM^2)$. Similarly, the complexity of $\nabla_W R$ is dominated by the multiplication of $\bar{H}_kW$, leading to $\mathcal{O}(KM^2K)$. For the sensing objective, the dominant operation in \eqref{eq:grad_w_crb} is the multiplication of $F^HMF$, which requires $\mathcal{O}(NM^2 + N^2M)$. Since the proposed framework alternates between the optimization of $\mathbf{F}$ and $\mathbf{W}$ over $I$ outer iterations and $\hat{J}$ adaptive inner iterations, the total computational complexity can be approximated as
%
\begin{equation}
\mathcal{O}\left(
I\hat{J}(KN^2M + KNM^2 + NM^2 + N^2M)
\right).
\end{equation}

Unlike conventional PGA-based methods that employ a fixed inner iteration count $J$, the proposed adaptive UPGA decay strategy dynamically adjusts the number of inner updates according to the gradient magnitude in \eqref{eq:calculate_J}. When the optimization approaches convergence, the gradient norm decreases, automatically reducing the required number of inner iterations. Consequently, unnecessary gradient computations are avoided, substantially lowering the computational burden. Experimental observations indicate that the proposed adaptive strategy reduces the average number of inner iterations by approximately $40\%$ compared to the fixed-step PGA approach while maintaining nearly identical communication and sensing performance. Therefore, the proposed UPGANet framework achieves a significantly improved complexity-performance tradeoff, making it suitable for practical large-scale ISAC systems.

\begin{table}[t]
\caption{Computational Complexity Comparison}
\label{tab:complexity_comparison}
\centering
\begin{tabular}{|c|c|c|}
\hline
\textbf{Method} & \textbf{Inner Iterations} & \textbf{Total Complexity} \\
\hline
Conventional PGA & $J$ &
$\mathcal{O}\left(IJ\Psi\right)$ \\
\hline
Proposed UPGANet & $\hat{J}$ &
$\mathcal{O}\left(I\hat{J}\Psi\right)$ \\
\hline
Complexity Reduction & $\approx 40\%$ &
$\approx 40\%$ lower \\
\hline
\end{tabular}
\vspace{1mm}

where
\[
\Psi = KN^2M + KNM^2 + NM^2 + N^2M.
\]
\end{table}

\section{Simulation Results}

In this section, simulation results are presented to evaluate the performance of the proposed UPGANet framework for hybrid beamforming in ISAC systems. We consider a mmWave MIMO-ISAC system where the BS is equipped with $N=64$ antennas and $M=4$ RF chains to simultaneously serve $K=4$ communication users while performing target sensing. The communication channels are generated using the Saleh-Valenzuela channel model with additive white Gaussian noise variance $\sigma_n^2$. The maximum transmit power is set to $P_{BS}=1$, and the signal-to-noise ratio (SNR) varies from $0$ dB to $12$ dB. For the optimization process, the maximum outer iteration number is selected as $I=120$, while different inner iteration settings are investigated for both the conventional PGA and the proposed unfolded PGA (UPGA) methods. Specifically, fixed inner iteration counts of $J=1$, $J=5$, and $J=10$ are considered together with the proposed adaptive decay strategy using $J_{\max}=5$ and $J_{\max}=10$. The performance is evaluated in terms of the objective function, achievable sum rate, and sensing accuracy characterized by the CRLB.

Fig.~(a) in \eqref{fig:objective_iterations} illustrates the convergence behavior of the objective function versus the outer iteration number. It can be observed that the proposed UPGA methods achieve faster convergence and higher objective values compared to the conventional PGA approaches. In particular, increasing the inner iteration count improves convergence performance; however, the proposed adaptive decay strategy achieves nearly identical performance while significantly reducing the required number of inner updates. Fig.~(b) demonstrates the achievable sum rate performance, where all unfolded approaches outperform the conventional PGA schemes, especially at higher iteration numbers. Similarly, Fig.~(c) shows the evolution of the sensing metric $I(\theta)$, indicating that the proposed method improves both communication and sensing performance simultaneously. Furthermore, the SNR performance comparison in Fig.~\eqref{fig:objective_snr} shows that the achievable sum rate increases with SNR, while the CRLB decreases due to improved sensing accuracy. Importantly, the adaptive UPGA decay methods provide performance very close to the full fixed-step unfolded PGA while reducing approximately $40\%$ of the inner iterations, thereby substantially lowering the overall computational complexity without noticeable performance degradation.

\begin{figure*}[!t]
    \centering
    \includegraphics[width=0.85\textwidth]{objective_vs_iterations.pdf}
    \caption{Performance comparison versus iteration number: (a) objective function, (b) achievable sum rate, and (c) sensing metric $I(\theta)$.}
    \label{fig:objective_iterations}
\end{figure*}

\begin{figure*}[!t]
    \centering
    \includegraphics[width=0.85\textwidth]{objective_vs_SNR.pdf}
    \caption{Performance comparison versus SNR: (a) achievable sum rate and (b) CRLB performance.}
    \label{fig:objective_snr}
\end{figure*}


@INPROCEEDINGS{10880627,
  author={Nguyen, Nhan Thanh and Nguyen, Ly V. and Shlezinger, Nir and Eldar, Yonina C. and Swindlehurst, A. Lee and Juntti, Markku},
  booktitle={2025 IEEE 5th International Symposium on Joint Communications & Sensing (JC&S)}, 
  title={Deep Unfolding-Empowered mmWave Massive MIMO Joint Communications and Sensing}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Runtime;Array signal processing;Massive MIMO;Transceivers;Sensors;Computational complexity;Optimization;Electronics packaging;Convergence;Joint communications and sensing;deep unfolding;hybrid beamforming},
  doi={10.1109/JCS64661.2025.10880627}}

@ARTICLE{9040264,
  author={Giordani, Marco and Polese, Michele and Mezzavilla, Marco and Rangan, Sundeep and Zorzi, Michele},
  journal={IEEE Communications Magazine}, 
  title={Toward 6G Networks: Use Cases and Technologies}, 
  year={2020},
  volume={58},
  number={3},
  pages={55-61},
  keywords={6G mobile communication;5G mobile communication;Reliability;Wireless networks;Internet of Things;Intelligent sensors},
  doi={10.1109/MCOM.001.1900411}}


@ARTICLE{8550811,
  author={Zhang, J. Andrew and Huang, Xiaojing and Guo, Y. Jay and Yuan, Jinhong and Heath, Robert W.},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Multibeam for Joint Communication and Radar Sensing Using Steerable Analog Antenna Arrays}, 
  year={2019},
  volume={68},
  number={1},
  pages={671-685},
  keywords={Sensors;Antenna arrays;Receivers;Radio transmitters;Baseband;Radar;Array signal processing;Joint communication and radar sensing;beamforming;multibeam;compressive sensing},
  doi={10.1109/TVT.2018.2883796}}


@ARTICLE{8030501,
  author={Molisch, Andreas F. and Ratnam, Vishnu V. and Han, Shengqian and Li, Zheda and Nguyen, Sinh Le Hong and Li, Linsheng and Haneda, Katsuyuki},
  journal={IEEE Communications Magazine}, 
  title={Hybrid Beamforming for Massive MIMO: A Survey}, 
  year={2017},
  volume={55},
  number={9},
  pages={134-141},
  keywords={Radio frequency;Array signal processing;MIMO communication;Antennas;NIST;Transceivers;Baseband},
  doi={10.1109/MCOM.2017.1600400}}


@ARTICLE{9729809,
  author={Qi, Chenhao and Ci, Wei and Zhang, Jinming and You, Xiaohu},
  journal={IEEE Communications Letters}, 
  title={Hybrid Beamforming for Millimeter Wave MIMO Integrated Sensing and Communications}, 
  year={2022},
  volume={26},
  number={5},
  pages={1136-1140},
  keywords={Radio frequency;MIMO communication;Array signal processing;Radar antennas;Signal to noise ratio;Interference;Sensors;Dual-functional radar-communication (DFRC);hybrid beamforming;integrated sensing and communications (ISAC);joint communications and radar (JCR);millimeter wave (mmWave) communications},
  doi={10.1109/LCOMM.2022.3157751}}


@ARTICLE{9868348,
  author={Wang, Xinyi and Fei, Zesong and Zhang, J. Andrew and Xu, Jie},
  journal={IEEE Transactions on Communications}, 
  title={Partially-Connected Hybrid Beamforming Design for Integrated Sensing and Communication Systems}, 
  year={2022},
  volume={70},
  number={10},
  pages={6648-6660},
  keywords={Sensors;Radio frequency;Signal to noise ratio;Interference;Array signal processing;Measurement;Radar;Integrated sensing and communications;hybrid analog-digital beamforming;optimization;Cramér-Rao bound},
  doi={10.1109/TCOMM.2022.3202215}}


@INPROCEEDINGS{9376319,
  author={Damith Liyanaarachchi, Sahan and Baquero Barneto, Carlos and Riihonen, Taneli and Heino, Mikko and Valkama, Mikko},
  booktitle={2021 1st IEEE International Online Symposium on Joint Communications & Sensing (JC&S)}, 
  title={Joint Multi-User Communication and MIMO Radar Through Full-Duplex Hybrid Beamforming}, 
  year={2021},
  volume={},
  number={},
  pages={1-5},
  keywords={Radio frequency;Array signal processing;MIMO radar;Superresolution;Estimation;Radar;Reflection},
  doi={10.1109/JCS52304.2021.9376319}}


@ARTICLE{9933894,
  author={Barneto, Carlos Baquero and Riihonen, Taneli and Liyanaarachchi, Sahan Damith and Heino, Mikko and González-Prelcic, Nuria and Valkama, Mikko},
  journal={IEEE Transactions on Communications}, 
  title={Beamformer Design and Optimization for Joint Communication and Full-Duplex Sensing at mm-Waves}, 
  year={2022},
  volume={70},
  number={12},
  pages={8298-8312},
  keywords={Sensors;Array signal processing;Radar;Radio frequency;Optimization;Antenna arrays;Reflection;Beamforming;full-duplex;joint communication and sensing;mm-wave;monostatic radar;self-interference},
  doi={10.1109/TCOMM.2022.3218623}}


@ARTICLE{9724206,
  author={Cheng, Ziyang and Liao, Bin},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={QoS-Aware Hybrid Beamforming and DOA Estimation in Multi-Carrier Dual-Function Radar-Communication Systems}, 
  year={2022},
  volume={40},
  number={6},
  pages={1890-1905},
  keywords={Radar;Radio frequency;Direction-of-arrival estimation;Quality of service;Estimation;Array signal processing;Interference;Hybrid beamforming (HBF);dual-function radar-communication (DFRC);QoS constraint;consensus-ADMM;DOA estimation},
  doi={10.1109/JSAC.2022.3155529}}


@ARTICLE{9366836,
  author={Cheng, Ziyang and He, Zishu and Liao, Bin},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Hybrid Beamforming for Multi-Carrier Dual-Function Radar-Communication System}, 
  year={2021},
  volume={7},
  number={3},
  pages={1002-1015},
  keywords={Radar;Radio frequency;Array signal processing;Radar detection;MIMO radar;MIMO communication;Conferences;Dual-function radar-communication (DFRC);hybrid beamfoming;multiple frequency signalings;subarray;weighted mean-square error minimization (WMMSE)},
  doi={10.1109/TCCN.2021.3063110}}


@INPROCEEDINGS{9909797,
  author={Wang, Bowen and Cheng, Ziyang and Wu, Linlong and He, Zishu},
  booktitle={2022 30th European Signal Processing Conference (EUSIPCO)}, 
  title={Hybrid Beamforming Design for OFDM Dual-Function Radar-Communication System with Double-Phase-Shifter Structure}, 
  year={2022},
  volume={},
  number={},
  pages={1067-1071},
  keywords={Array signal processing;Spectral efficiency;OFDM;Simulation;Spaceborne radar;Signal processing algorithms;Interference;Hybrid beamforming (HBF);mmWave;OFDM-DFRC;double-phase-shifter (DPS)},
  doi={10.23919/EUSIPCO55093.2022.9909797}}


@INPROCEEDINGS{9838368,
  author={Islam, Md Atiqul and Alexandropoulos, George C. and Smida, Besma},
  booktitle={ICC 2022 - IEEE International Conference on Communications}, 
  title={Integrated Sensing and Communication with Millimeter Wave Full Duplex Hybrid Beamforming}, 
  year={2022},
  volume={},
  number={},
  pages={4673-4678},
  keywords={Direction-of-arrival estimation;Array signal processing;OFDM;Estimation;Millimeter wave radar;Sensors;Doppler radar;Full duplex;millimeter wave;direction estimation;range-Doppler estimation;joint communication and sensing},
  doi={10.1109/ICC45855.2022.9838368}}


@INPROCEEDINGS{9562238,
  author={Elbir, Ahmet M. and Mishra, Kumar Vjiay and Chatzinotas, Symeon},
  booktitle={2021 17th International Symposium on Wireless Communication Systems (ISWCS)}, 
  title={Hybrid Beamforming for Terahertz Joint Ultra-Massive MIMO Radar-Communications}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  keywords={Wireless communication;Costs;Array signal processing;Spectral efficiency;Radar;Radar antennas;Propagation losses;Joint radar-communications;terahertz;hybrid beamforming;ultramassive MIMO},
  doi={10.1109/ISWCS49558.2021.9562238}}


@ARTICLE{10539066,
  author={Leyva, Leonardo and Castanheira, Daniel and Silva, Adão and Gameiro, Atílio},
  journal={IEEE Sensors Journal}, 
  title={Hybrid Beamforming Design for Communication-Centric ISAC}, 
  year={2024},
  volume={24},
  number={13},
  pages={21179-21190},
  keywords={Array signal processing;Sensors;Optimization;Signal to noise ratio;Radio frequency;Vectors;Complexity theory;Fully connected hybrid beamforming;integrated sensing and communication (ISAC);iterative alternate optimization;multibeam steering;multiuser communication;radio sensing},
  doi={10.1109/JSEN.2024.3403032}}


@ARTICLE{10540175,
  author={Liu, Wanxian and Xu, Hongbo and He, Xiuli and Ye, Yuchen and Zhou, Aizhi},
  journal={IEEE Access}, 
  title={Bi-Level Deep Unfolding Based Robust Beamforming Design for IRS-Assisted ISAC System}, 
  year={2024},
  volume={12},
  number={},
  pages={76663-76672},
  keywords={Sensors;Signal to noise ratio;Interference;Array signal processing;Uncertainty;Power system reliability;Linear programming;Communication systems;Optimization methods;Power system reliability;Integrated sensing and communication system;intelligent reflecting surface;robust beamforming;bi-level optimization;deep unfolding;worst-case CRB;SINR outage probability},
  doi={10.1109/ACCESS.2024.3406527}}

@ARTICLE{9830727,
  author={Kang, Kai and Hu, Qiyu and Cai, Yunlong and Yu, Guanding and Hoydis, Jakob and Eldar, Yonina C.},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Mixed-Timescale Deep-Unfolding for Joint Channel Estimation and Hybrid Beamforming}, 
  year={2022},
  volume={40},
  number={9},
  pages={2510-2528},
  keywords={Array signal processing;Radio frequency;Channel estimation;Artificial neural networks;Massive MIMO;Computational complexity;Training;Deep-unfolding;hybrid beamforming;channel estimation;mixed-timescale scheme;massive MIMO},
  doi={10.1109/JSAC.2022.3191124}}


