"""PredictiveMicrocircuits — predictive coding with bidirectional error signals.

Biological basis (Friston 2005; Bastos et al. 2012):
  Layer 2/3 error neurons encode SIGNED prediction error:
    - Positive error (input > prediction): "more than expected"
    - Negative error (input < prediction): "less than expected"
  Both populations exist in V1 (Keller & Mrsic-Flogel 2018).

  Top-down weights learn to minimise prediction error (predictive).
  Bottom-up weights learn to propagate residual error upward (inferential).
  These are DIFFERENT learning rules — not the same Hebbian update.

  Surprise is normalised by active node count so it stays comparable
  across networks of different sizes.

  Weights clamped to [-1, 1] to prevent unbounded accumulation
  (mentioned as known sharp edge in CLAUDE.md).
"""

import numpy as np
import torch


class PredictiveMicrocircuits:

    def __init__(
        self,
        num_nodes: int,
        initial_density: float = 0.005,
        device: torch.device | None = None,
        max_fan_in: int = 64,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = num_nodes
        dense_target = int(num_nodes * num_nodes * initial_density)
        capped_target = int(num_nodes * max_fan_in)
        num_connections = min(dense_target, capped_target)

        # Predictive microcircuits are sparse local motifs, not a second full-brain
        # all-to-all graph. Cap fan-in per neuron so startup memory scales linearly.
        if dense_target > capped_target:
            print(f'    [БИОЛОГИЯ] PredictiveMicrocircuits capped: '
                  f'{dense_target:,} -> {num_connections:,} edges '
                  f'(~{max_fan_in} / neuron)')

        # Coalesce duplicate (src, tgt) at init so values stay aligned with
        # _W_top sparse tensor (which always coalesces internally).
        def _build(n):
            # Generate and deduplicate on CPU via numpy (lower peak RAM than torch)
            rng = np.random.default_rng()
            idx_np = rng.integers(0, num_nodes, size=(2, n), dtype=np.int64)
            keys = idx_np[0] * num_nodes + idx_np[1]
            _, unique_idx = np.unique(keys, return_index=True)
            unique_idx.sort()  # preserve original order among uniques
            deduped = idx_np[:, unique_idx]
            # Sort by (src, tgt) for coalesced sparse tensor compatibility
            sort_keys = deduped[0] * num_nodes + deduped[1]
            sort_order = np.argsort(sort_keys)
            sorted_idx = deduped[:, sort_order]
            vals = torch.rand(sorted_idx.shape[1], device=self.device) * 0.1
            return torch.from_numpy(sorted_idx.copy()).to(self.device), vals

        self.td_indices, self.td_values = _build(num_connections)
        self.bu_indices, self.bu_values = _build(num_connections)

        self.prediction_neurons  = torch.zeros(num_nodes, device=self.device)
        self.error_neurons       = torch.zeros(num_nodes, device=self.device)

        self._W_top: torch.Tensor | None = None
        self._W_bot: torch.Tensor | None = None

    def process_inference_step(
        self,
        sensory_input:    torch.Tensor,
        internal_state:   torch.Tensor,
        plasticity_rate:  float,
    ) -> tuple:
        # ------------------------------------------------------------------
        # 1. Top-down prediction (rebuild sparse tensor only when weights changed)
        # ------------------------------------------------------------------
        dev = sensory_input.device
        if self._W_top is None:
            self._W_top = torch.sparse_coo_tensor(
                self.td_indices, self.td_values,
                (self.num_nodes, self.num_nodes)).coalesce().to(dev)
        else:
            self._W_top.values().copy_(self.td_values.to(dev))
        W_top = self._W_top
        self.prediction_neurons = torch.clamp(
            torch.sparse.mm(W_top, internal_state.unsqueeze(1)).squeeze(1),
            0.0, 1.0)

        # ------------------------------------------------------------------
        # 2. Bidirectional prediction error (signed, not relu)
        #    Positive:  input > prediction → "more than expected" (L2/3 ON cells)
        #    Negative:  input < prediction → "less than expected" (L2/3 OFF cells)
        # ------------------------------------------------------------------
        self.error_neurons = sensory_input - self.prediction_neurons   # signed [-1, 1]

        error_abs = torch.abs(self.error_neurons)

        # ------------------------------------------------------------------
        # 3. Bottom-up drive: propagate signed error to update internal state
        # ------------------------------------------------------------------
        if self._W_bot is None:
            self._W_bot = torch.sparse_coo_tensor(
                self.bu_indices, self.bu_values,
                (self.num_nodes, self.num_nodes)).coalesce().to(dev)
        else:
            self._W_bot.values().copy_(self.bu_values.to(dev))
        W_bot = self._W_bot
        bottom_up_drive = torch.sparse.mm(
            W_bot, self.error_neurons.unsqueeze(1)).squeeze(1)
        updated_internal_state = torch.clamp(
            internal_state + bottom_up_drive, 0.0, 1.0)

        # ------------------------------------------------------------------
        # 4. Weight learning — DIFFERENT rules for TD vs BU
        #
        # TD (predictive):  weights grow when prediction co-active with error,
        #   driving predictions toward reducing the error (prediction-error
        #   minimisation, Rao & Ballard 1999).
        #   Δw_td = α × |error_post| × (state_pre > 0)
        #
        # BU (inferential): weights grow when error nodes co-activate, encoding
        #   how to propagate residual error upward to higher levels.
        #   Δw_bu = α × |error_pre| × |error_post|
        # ------------------------------------------------------------------
        alpha = 0.005 * plasticity_rate

        # TD: prediction neurons (pre=internal state) → error nodes (post)
        active_state = internal_state > 0.1
        td_active    = active_state[self.td_indices[1]] & (error_abs[self.td_indices[0]] > 0.05)
        if td_active.any():
            self.td_values[td_active] = torch.clamp(
                self.td_values[td_active]
                + alpha * error_abs[self.td_indices[0]][td_active],
                0.0, 1.0)

        # BU: error source (pre) → error destination (post): propagate surprise upward
        bu_active = (error_abs[self.bu_indices[0]] > 0.05) & (error_abs[self.bu_indices[1]] > 0.05)
        if bu_active.any():
            self.bu_values[bu_active] = torch.clamp(
                self.bu_values[bu_active]
                + alpha * error_abs[self.bu_indices[0]][bu_active] * error_abs[self.bu_indices[1]][bu_active],
                0.0, 1.0)

        # ------------------------------------------------------------------
        # 5. Normalised surprise (per active node, size-invariant)
        # ------------------------------------------------------------------
        n_active = float(max(1, (sensory_input > 0.1).sum().item()))
        total_surprise = float(error_abs.sum()) / n_active

        return total_surprise, updated_internal_state
