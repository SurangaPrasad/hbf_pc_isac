"""SelectionNet: learnable antenna-to-RF-chain assignment for sub-connected hybrid beamforming.

In a sub-connected mmWave massive MIMO ISAC system with N_t transmit antennas and
N_rf RF chains (N_t > N_rf), every antenna must be hard-wired to exactly one RF
chain, and the network learns a connection matrix S in (N_t, N_rf) that tells the
hybrid precoder which antennas are driven by which RF chain.  Because there is no
ground-truth assignment, S is trained with an unsupervised (downstream physics)
objective, so this module only needs to produce a differentiable, row-stochastic
connection matrix from the channel H and a sensing direction psi0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelectionNet(nn.Module):
    """MLP that maps (channel, sensing direction) -> row-stochastic connection matrix S.

    Architecture summary
    --------------------
    H (complex) is turned into a real feature stack [real, imag, |H|] and flattened;
    psi0 is encoded as [sin, cos]; the two are concatenated and fed through a 2-layer
    ReLU MLP whose output is reshaped into per-antenna logits of shape (B, N_t, N_rf).
    A row-wise Gumbel-softmax (implemented from scratch, straight-through for hard
    mode) turns those logits into a differentiable, row-stochastic S.

    Parameters
    ----------
    n_antennas : int
        Number of transmit antennas N_t.
    n_rf_chains : int
        Number of RF chains N_rf (equals the connection-matrix width M).
    n_users : int
        Number of served users K (width of the channel matrix H).
    hidden_dim : int, default 128
        Width of the two hidden MLP layers.

    Note
    ----
    The encoder input dimension is ``3 * N_t * K + 2`` (real/imag/mag features of H
    plus the two sin/cos components of psi0), so it is a function of N_t, K and is
    fixed at construction time.
    """

    def __init__(self, n_antennas: int, n_rf_chains: int, n_users: int, hidden_dim: int = 128) -> None:
        super().__init__()

        # Keep the sizes around so the model knows its own output geometry.
        self.n_antennas = n_antennas
        self.n_rf_chains = n_rf_chains
        self.n_users = n_users

        # Flattened H contributes 3 * N_t * K real features (real/imag/magnitude per
        # complex entry); psi0 contributes 2 (sin, cos). Together they fix the MLP
        # input width.
        h_features = 3 * n_antennas * n_users
        encoder_in = h_features + 2

        # 2 hidden layers + ReLU. Kept as separate nn.Sequential layers so the
        # forward pass reads like the architecture diagram in the paper notes.
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # One scalar logit per (antenna, RF chain) pair. No activation here: softmax
        # lives downstream and the raw logits are also exposed to the caller.
        self.logits_head = nn.Linear(hidden_dim, n_antennas * n_rf_chains)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _encode_psi0(psi0: torch.Tensor) -> torch.Tensor:
        """Encode a scalar angle as [sin(psi0), cos(psi0)] -> (B, 2).

        Why sin/cos instead of the raw radian value? Angles are periodic, so a plain
        scalar would place psi0 = 0 and psi0 = 2*pi "far apart" in feature space even
        though they are physically the same direction. Mapping onto the unit circle
        makes the encoding smooth and wrap-around safe, so nearby steering directions
        stay nearby in feature space.
        """
        return torch.stack([torch.sin(psi0), torch.cos(psi0)], dim=-1)

    @staticmethod
    def _encode_channel(H: torch.Tensor) -> torch.Tensor:
        """Convert complex H (B, N_t, K) into a real feature stack (B, 3, N_t, K).

        nn.Linear only works on real tensors, so a complex channel must be split into
        real-valued channels first. Real and imaginary parts preserve the full complex
        information; appending the magnitude gives the network a rotation-invariant
        summary of each channel entry that is cheap to read off directly.
        """
        real = H.real
        imag = H.imag
        mag = H.abs()
        # Stack along a new channel dim (dim=1), giving (B, 3, N_t, K). The caller
        # then flattens this; keeping the 3 as a leading "channel" axis documents the
        # three complementary views before collapsing them for the MLP.
        return torch.stack([real, imag, mag], dim=1)

    def gumbel_softmax(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """Row-wise Gumbel-softmax over the last (RF-chain) dimension.

        Implements the differentiable approximation to discrete sampling described in
        Jang et al. 2016 / Maddison et al. 2016, hand-rolled so the mechanics are
        transparent (a later swap to ``torch.nn.functional.gumbel_softmax`` or a
        learned temperature would be a one-line change in ``forward``).

        Why Gumbel noise at all? ``argmax`` is not differentiable, and ``softmax``
        without noise converges to a deterministic (non-exploring) policy. Adding
        Gumbel noise perturbs each logit so the softmax becomes a *differentiable
        sample* from the categorical distribution: modes are still favored, but the
        network keeps exploring during training.

        Why row-wise over the RF-chain dimension? The sub-connected constraint is a
        per-antenna choice — each of the N_t antennas picks exactly one of the N_rf
        RF chains — so the categorical distribution is over the M=N_rf options for
        every antenna row independently. Row-wise softmax makes each row sum to 1
        automatically, which is exactly the "one RF chain per antenna" hard constraint
        in its relaxed form.

        Straight-through estimator (hard=True): during the forward pass we return the
        true one-hot argmax so the actual assignment is discrete; on the backward pass
        we pretend the identity of the argmax doesn't matter and let gradients flow
        through the soft probabilities instead (``hard - soft.detach() + soft``).
        ``soft.detach()`` blocks gradients on the hard part, ``+ soft`` re-injects the
        real soft gradient, so the resulting tensor equals hard one-hot in value yet
        receives soft-probability gradients.

        Parameters
        ----------
        logits : torch.Tensor
            Pre-softmax logits, shape (B, N_t, N_rf).
        tau : float, default 1.0
            Temperature. tau -> 0 sharpens toward one-hot; larger tau smooths and
            increases exploration. Usually annealed from high to low over training.
        hard : bool, default False
            If True, return discrete one-hot rows with straight-through gradients.

        Returns
        -------
        torch.Tensor
            Shape (B, N_t, N_rf), rows sum to 1.
        """
        # Gumbel noise from uniform samples: G = -log(-log(U)), U ~ Uniform(0, 1).
        # The 1e-20 epsilons guard against log(0) / division by zero in edge cases.
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform.clamp_min(1e-20)) + 1e-20)

        # Perturbed logits, scaled by the temperature, then a plain softmax over the
        # RF-chain axis (dim=-1). This is the differentiable sample.
        perturbed = (logits + gumbel_noise) / tau
        soft = torch.softmax(perturbed, dim=-1)

        if not hard:
            return soft

        # One-hot by taking the argmax of the (monotone-transformed) soft scores.
        # scatter_ along the last dim writes a 1.0 at each row's winning column.
        hard_onehot = torch.zeros_like(soft)
        winners = soft.argmax(dim=-1, keepdim=True)
        hard_onehot.scatter_(-1, winners, 1.0)

        # Straight-through trick: value equals the hard one-hot, gradient equals the
        # soft gradient.
        return hard_onehot - soft.detach() + soft

    # ------------------------------------------------------------------ public
    def forward(
        self,
        H: torch.Tensor,
        psi0: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a channel and a sensing direction to a connection matrix S.

        Parameters
        ----------
        H : torch.Tensor
            Complex channel matrix, shape (batch, N_t, K).
        psi0 : torch.Tensor
            Sensing target direction in radians, shape (batch,).
        tau : float, default 1.0
            Gumbel-softmax temperature passed to ``gumbel_softmax``.
        hard : bool, default False
            Straight-through hard one-hot mode for the returned S.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            S : (batch, N_t, N_rf) connection matrix, rows sum to 1.
            logits : (batch, N_t, N_rf) raw pre-softmax logits (also used to build
            S), exposed for inspection or a logits-level regularizer elsewhere.
        """
        batch = H.shape[0]

        # (B, 3, N_t, K) -> flatten to (B, 3*N_t*K) for the MLP. Flattening keeps
        # this an MLP rather than a conv net; ordering within the feature vector is
        # irrelevant to the MLP since every entry connects to every hidden unit.
        h_features = self._encode_channel(H).reshape(batch, -1)

        psi_features = self._encode_psi0(psi0)  # (B, 2)

        # Concatenate along the feature dimension and push through the MLP.
        features = torch.cat([h_features, psi_features], dim=-1)
        hidden = self.encoder(features)

        # (B, N_t*N_rf) -> (B, N_t, N_rf): one logit per (antenna, RF chain) entry.
        logits = self.logits_head(hidden).view(batch, self.n_antennas, self.n_rf_chains)

        # Row-wise Gumbel-softmax over the RF-chain axis -> differentiable S.
        S = self.gumbel_softmax(logits, tau=tau, hard=hard)

        return S, logits

    def column_load(self, S: torch.Tensor) -> torch.Tensor:
        """Per-RF-chain antenna load: sum of S over the antenna axis (dim=1).

        Returns shape (batch, N_rf): entry [b, m] is how many antennas RF chain m
        serves (fractional counts when S is soft, integer counts when S is hard).
        Designed as a standalone utility so a load-balancing regularizer can be
        added downstream without touching this module.
        """
        return S.sum(dim=1)


if __name__ == "__main__":
    # Minimal smoke test: verify shapes and the row-stochastic constraint in both
    # soft and hard modes. Kept tiny (batch 4, 64 antennas, 4 users) so it runs
    # anywhere without a GPU.
    torch.manual_seed(0)

    batch, n_t, n_rf, n_users = 4, 64, 4, 4

    # Random complex channel H and random sensing angles in [-pi/2, pi/2].
    H = torch.randn(batch, n_t, n_users, dtype=torch.complex64)
    psi0 = (torch.rand(batch) - 0.5) * math.pi

    net = SelectionNet(n_antennas=n_t, n_rf_chains=n_rf, n_users=n_users)

    S_soft, logits_soft = net(H, psi0, tau=1.0, hard=False)
    print(f"soft  mode: S={tuple(S_soft.shape)}, logits={tuple(logits_soft.shape)}")

    S_hard, logits_hard = net(H, psi0, tau=1.0, hard=True)
    print(f"hard  mode: S={tuple(S_hard.shape)}, logits={tuple(logits_hard.shape)}")

    # Row-wise softmax guarantees every row sums to 1.0 in both modes; assert with a
    # loose tolerance that still catches a broken axis (e.g. softmax over dim=0).
    for name, S in (("soft", S_soft), ("hard", S_hard)):
        row_sums = S.sum(dim=-1)
        lo, hi = row_sums.min().item(), row_sums.max().item()
        print(f"{name} mode row sums: min={lo:.6f}, max={hi:.6f}")
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"{name} rows do not sum to 1"

    # column_load is just a dim-1 sum; print its shape as a sanity check.
    loads = net.column_load(S_soft)
    print(f"column_load(S_soft): {tuple(loads.shape)}, first row: {loads[0].tolist()}")

    print("All assertions passed.")
