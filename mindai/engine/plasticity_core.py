"""StructuralPlasticity — Hebbian/STDP learning with biological accuracy.

STDP asymmetry (Bi & Poo 1998):
  Causal   (pre→post, LTP): τ+ ≈ 20 ms,  A+ = 0.05   (faster trace, larger)
  Anti-causal (post→pre, LTD): τ- ≈ 40 ms,  A- = 0.015  (slower trace, smaller)
  Implemented via different decay rates: pre_trace *= 0.90, post_trace *= 0.85

Sign convention:
  Inhibitory pre-synaptic neurons carry negative weights.  STDP strengthens
  them in the negative direction (more inhibitory) and weakens in the positive
  direction — consistent with the receptor type at the post-synaptic membrane.
  LTP on an inhibitory synapse → weight more negative → correct biology.
  LTD on an inhibitory synapse → weight less negative → correct biology.

Homeostatic synaptic scaling (Turrigiano 1998):
  Target incoming weight sum scales with network density so it stays valid
  across any num_neurons / density combination.

Critical-period plasticity (Hubel & Wiesel 1970; Hensch 2005):
  Heightened plasticity early in life, gradually closing as the brain ages.
  Implemented via `critical_period_factor`: STDP rate is multiplied by
  this factor, which decays from ~3.0 toward 1.0 over the first ~50k ticks
  (≈ infant-to-adolescent maturation curve at 10 Hz). Older brains can
  still learn but at adult rate.
"""

import torch
import numpy as np
import random


class StructuralPlasticity:

    def __init__(
        self,
        num_nodes:        int,
        initial_density:  float = 0.01,
        inhibitory_ratio: float = 0.2,
        device:           torch.device | None = None,
        coordinates:      np.ndarray | None = None,
    ):
        self.device           = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes        = num_nodes
        # Neonatal cortex: ~20% of neurons active at birth, matures to ~80% max.
        # Neurogenesis (trigger_neurogenesis) gradually unlocks the rest.
        # Huttenlocher 1979: synaptic density peaks ~2 years then prunes to adult level.
        self.active_limit     = max(int(num_nodes * 0.20), 1000)
        self._active_ceiling  = int(num_nodes * 0.80)   # adult maximum
        self.epistemic_hunger = 0.0
        self.hunger_threshold = 100.0
        self.growth_cooldown  = 0
        self._initial_density = initial_density

        # Homeostatic target: mean incoming weight sum per neuron.
        # At density d, each neuron has ~d*N inputs; target keeps mean weight ~0.5.
        # Scales proportionally so it stays valid across network sizes.
        self._homeostatic_target = max(2.0, initial_density * self.active_limit * 0.5)

        print(f'    [БИОЛОГИЯ] Выращивание графа на {self.device}. '
              f'80% Глутамат / 20% ГАМК...')

        # Load coordinates
        if coordinates is not None:
            self.coordinates = torch.from_numpy(coordinates).to(self.device)
        else:
            # Fallback coordinate generation
            idx = np.arange(num_nodes, dtype=np.float64)
            phi = np.arccos(1.0 - 2.0 * (idx + 0.5) / num_nodes)
            theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
            sin_phi = np.sin(phi)
            coords = np.empty((num_nodes, 3), dtype=np.float32)
            coords[:, 0] = 10.0 * np.cos(theta) * sin_phi
            coords[:, 1] = 10.0 * np.sin(theta) * sin_phi
            coords[:, 2] = 10.0 * np.cos(phi)
            # Sort fallback coordinates by Morton code
            from mindai.engine.spatial_topology_3d import get_morton_codes_np
            morton_codes = get_morton_codes_np(coords)
            sort_idx = np.argsort(morton_codes)
            coords = coords[sort_idx]
            self.coordinates = torch.from_numpy(coords).to(self.device)

        self.is_inhibitory_tensor = torch.rand(num_nodes, device=self.device) < inhibitory_ratio
        self.is_inhibitory        = self.is_inhibitory_tensor.cpu().numpy()

        num_connections = int(self.active_limit * self.active_limit * initial_density)
        # Use physical proximity to establish initial synapses
        src_c = torch.arange(self.active_limit, device=self.device)
        src, tgt = self._sample_proximity_pairs(src_c, src_c, num_connections, sigma=2.0)
        mask    = src != tgt
        self.indices = torch.stack([src[mask], tgt[mask]])

        signs = torch.where(self.is_inhibitory_tensor[self.indices[0]], -1.0, 1.0)
        self.weights_values   = (torch.rand(self.indices.shape[1], device=self.device) * 0.09 + 0.01) * signs
        self.integrity_values = torch.ones(self.indices.shape[1], device=self.device)

        # Dopamine eligibility trace per synapse
        self.eligibility = torch.zeros(self.indices.shape[1], device=self.device)

        # LIF Membrane potential and threshold parameters
        self.v_mem = torch.zeros(num_nodes, device=self.device)
        self.firing_thresholds = torch.ones(num_nodes, device=self.device)
        self.firing_rates = torch.full((num_nodes,), 0.05, device=self.device)
        self.v_reset = 0.0
        self.v_leak = 0.8

        # STDP traces — DIFFERENT decay rates to implement temporal asymmetry
        # pre_trace  (for LTP): τ+ ≈ 20 ms → decay 0.85 at 10 Hz (narrower causal window)
        # post_trace (for LTD): τ- ≈ 40 ms → decay 0.90 at 10 Hz (wider anti-causal window)
        # Bi & Poo 1998: τ- > τ+ — LTD window must be wider than LTP window.
        self.pre_trace  = torch.zeros(num_nodes, device=self.device)
        self.post_trace = torch.zeros(num_nodes, device=self.device)

        # Refractory period — Na⁺ channel inactivation (Hodgkin & Huxley 1952)
        # After a spike the neuron cannot fire for ~2 ms (absolute refractory).
        # Stored as a countdown tensor: >0 means in refractory, decremented each tick.
        # 2 ticks at 10 Hz ≈ 2 ms biological.
        self._refractory = torch.zeros(num_nodes, dtype=torch.int16, device=self.device)
        self._REFRACTORY_TICKS = 2

        # Spike-frequency adaptation — slow K⁺ channel activation (Bhattacharjee 2005)
        # Prolonged firing opens K+ (KCa, Kv7) channels → hyperpolarisation → lower
        # effective excitability. Modelled as an adaptation current that accumulates
        # when the neuron fires and decays when it is silent (τ ≈ 100 ms → 0.90/tick).
        self._adaptation = torch.zeros(num_nodes, device=self.device)

        # Short-term synaptic plasticity — Tsodyks & Markram 1997
        # u: utilization (facilitation variable, increases with each spike)
        # x: available vesicle fraction (depression variable, decreases with use)
        # Parameters: τ_rec = 800 ms (0.988/tick), τ_fac = 500 ms (0.982/tick)
        # Facilitating synapses (inh→exc) vs depressing synapses (exc→exc) differ
        # in U₀: here uniform U₀=0.3 for simplicity (Zucker & Regehr 2002).
        n_syn = self.indices.shape[1]
        self._stp_u = torch.full((n_syn,), 0.3, device=self.device)  # utilization
        self._stp_x = torch.ones(n_syn,        device=self.device)  # vesicle fraction

        self._cached_sparse_weights = None
        self._cached_sparse_weights_stp = None
        self._topology_changed      = True

        # Critical-period plasticity (Hensch 2005)
        # Brain "age" in ticks; STDP rate multiplied by critical_period_factor
        # which starts at 3.0 and decays toward 1.0 over ~50k ticks.
        self._age_ticks: int = 0
        self._cp_initial = 3.0
        self._cp_tau     = 50_000.0    # ticks to ~e-fold decay

    # ------------------------------------------------------------------
    # Single-neuron dynamics — refractory + adaptation (applied before STDP)
    # ------------------------------------------------------------------

    def apply_neural_dynamics(self, activity: torch.Tensor, acetylcholine: float = 0.5) -> torch.Tensor:
        """Enforce refractory period, spike-frequency adaptation, and LIF integration.

        Integrates incoming activation (current) into membrane potential v_mem,
        generating a binary spike when threshold is reached, and then
        resetting membrane potential and applying refractory periods.

        Refractory: Hodgkin & Huxley 1952 — absolute refractory 2 ticks.
        Adaptation:  Bhattacharjee & Bhattacharjee 2005 — slow K⁺ activation.
        LIF integration: leaky integrate-and-fire spike threshold.
        """
        refractory_mask = self._refractory > 0

        # Adaptation gates effective input current gain: heavy firing -> lower gain
        adaptation_gain = torch.clamp(1.0 - self._adaptation * 0.7, 0.3, 1.0)
        input_current = activity * adaptation_gain * (~refractory_mask).float()

        # Somatic lateral feedback inhibition (GABAergic basket cell feedback)
        # Total excitation of the excitatory population is summed
        exc_mask = ~self.is_inhibitory_tensor
        total_exc = (input_current * exc_mask.float()).sum()

        # Acetylcholine modulates inhibition: high ACh (attention) -> less inhibition
        # We apply feedback inhibition to suppress weaker signals and maintain sparsity
        inh_gain = 0.02 * (1.0 - acetylcholine * 0.4)
        feedback_inh = total_exc * inh_gain

        # Integrate input current and apply feedforward somatic inhibition to excitatory neurons
        self.v_mem += input_current
        self.v_mem[exc_mask] = torch.clamp(self.v_mem[exc_mask] - feedback_inh, min=0.0)

        # Detect spikes using individual dynamic homeostatic thresholds
        fired_now = self.v_mem >= self.firing_thresholds

        # Reset membrane potential and trigger refractory for winners
        self.v_mem[fired_now] = self.v_reset
        self._refractory[fired_now] = self._REFRACTORY_TICKS

        # Decrement refractory countdown for others
        self._refractory[~fired_now] = torch.clamp(
            self._refractory[~fired_now] - 1, min=0).to(torch.int16)

        # Leak membrane potential for non-firing neurons
        self.v_mem[~fired_now] *= self.v_leak

        # Update adaptation (slow K⁺ current accumulation, τ ≈ 100 ms -> decay 0.90/tick)
        self._adaptation *= 0.90
        self._adaptation[fired_now] = torch.clamp(
            self._adaptation[fired_now] + 0.15, 0.0, 1.0)

        # Intrinsic Excitability Homeostasis:
        # Firing rate estimate (exponential moving average, alpha=0.01)
        self.firing_rates = (1.0 - 0.01) * self.firing_rates + 0.01 * fired_now.float()

        # Adjust thresholds: increase if firing rate is too high, decrease if too low (silent)
        # Target sparsity is 0.05. beta = 0.02. Range clamped to biological [0.2, 5.0]
        self.firing_thresholds = torch.clamp(
            self.firing_thresholds + 0.02 * (self.firing_rates - 0.05),
            min=0.2, max=5.0
        )

        return fired_now.float()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _coalesce_state(self):
        # Coalesce all per-edge state together so duplicates get merged consistently
        combined = torch.stack([self.weights_values, self.integrity_values,
                                self._stp_u, self._stp_x, self.eligibility], dim=1)
        temp = torch.sparse_coo_tensor(
            self.indices, combined,
            (self.num_nodes, self.num_nodes, 5)).coalesce()
        self.indices          = temp.indices()
        v                     = temp.values()
        self.weights_values   = v[:, 0]
        self.integrity_values = torch.clamp(v[:, 1], 0.0, 2.0)
        self._stp_u           = torch.clamp(v[:, 2], 0.0, 1.0)
        self._stp_x           = torch.clamp(v[:, 3], 0.0, 1.0)
        self.eligibility      = v[:, 4]
        coo = torch.sparse_coo_tensor(
            torch.stack([self.indices[1], self.indices[0]]), self.weights_values, (self.num_nodes, self.num_nodes)).coalesce()
        self._cached_sparse_weights = coo.to_sparse_csr()
        self._topology_changed = False

        # Compute permutation mapping for transposed sparse matrix representation (sorted indices[1], indices[0])
        # This handles the index swap sorting correct mapping when copying values back.
        keys = self.indices[1] * self.num_nodes + self.indices[0]
        self._transpose_perm = torch.argsort(keys)

    def get_sparse_weights(self, apply_stp: bool = False) -> torch.Tensor:
        """Return cached sparse weight matrix, optionally scaled by STP efficacy.

        apply_stp=True: weights multiplied by u*x (release probability × vesicles).
        Called with apply_stp=True each tick from brain.py; False during sleep replay
        where STP state should not be consumed.
        """
        if getattr(self, '_topology_changed', True) or self._cached_sparse_weights is None:
            self._coalesce_state()
        else:
            self._cached_sparse_weights.values().copy_(self.weights_values[self._transpose_perm])

        if apply_stp:
            stp_efficacy = self._stp_u * self._stp_x   # release probability × resource
            eff_vals = self.weights_values * stp_efficacy
            if getattr(self, '_cached_sparse_weights_stp', None) is None or self._topology_changed:
                coo_stp = torch.sparse_coo_tensor(
                    torch.stack([self.indices[1], self.indices[0]]), eff_vals, (self.num_nodes, self.num_nodes)).coalesce()
                self._cached_sparse_weights_stp = coo_stp.to_sparse_csr()
            else:
                self._cached_sparse_weights_stp.values().copy_(eff_vals[self._transpose_perm])
            return self._cached_sparse_weights_stp

        return self._cached_sparse_weights

    def get_stp_scaled_weights(self) -> torch.Tensor:
        """Return raw flat weights scaled by current STP efficacy (release probability * resources)."""
        return self.weights_values * self._stp_u * self._stp_x

    def step_stp(self, activity: torch.Tensor) -> None:
        """Advance short-term synaptic plasticity state for one tick.

        Tsodyks & Markram 1997:
          u += U₀ × (1 − u)    when pre fires  (facilitation: Ca²⁺ → higher release P)
          x -= u × x            when pre fires  (depression: vesicle depletion)
          u recovers: τ_fac = 500 ms → 0.982/tick
          x recovers: τ_rec = 800 ms → 0.988/tick
        """
        self._stp_u *= 0.982   # τ_fac recovery
        self._stp_x *= 0.988   # τ_rec recovery — vesicle replenishment
        self._stp_x = torch.clamp(self._stp_x, 0.0, 1.0)

        pre_fired = activity[self.indices[0]] > 0.5
        if pre_fired.any():
            U0 = 0.3
            # Facilitation: Ca²⁺ influx increases vesicle release probability
            self._stp_u[pre_fired] = torch.clamp(
                self._stp_u[pre_fired] + U0 * (1.0 - self._stp_u[pre_fired]), 0.0, 1.0)
            # Depression: released vesicles temporarily depleted
            self._stp_x[pre_fired] = torch.clamp(
                self._stp_x[pre_fired] - self._stp_u[pre_fired] * self._stp_x[pre_fired],
                0.0, 1.0)

    # ------------------------------------------------------------------
    # Neurogenesis — van Praag 2002; Eriksson 1998
    # ------------------------------------------------------------------

    def _sample_proximity_pairs_chunk(
        self,
        src_candidates: torch.Tensor,
        tgt_candidates: torch.Tensor,
        n_pairs:        int,
        sigma:          float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(src_candidates) == 0 or len(tgt_candidates) == 0:
            return (torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))

        if len(src_candidates) == 1:
            src = src_candidates.expand(n_pairs)
        else:
            src_idxs = torch.randint(0, len(src_candidates), (n_pairs,), device=self.device)
            src = src_candidates[src_idxs]

        src_coords = self.coordinates[src]
        perturbed = src_coords + torch.randn_like(src_coords) * sigma

        if len(tgt_candidates) == 1:
            tgt = tgt_candidates.expand(n_pairs)
        else:
            k_candidates = min(8, len(tgt_candidates))
            tgt_pool_idxs = torch.randint(0, len(tgt_candidates), (n_pairs, k_candidates), device=self.device)
            tgt_pool = tgt_candidates[tgt_pool_idxs]

            tgt_coords = self.coordinates[tgt_pool]
            diffs = tgt_coords - perturbed.unsqueeze(1)
            dists = torch.sum(diffs ** 2, dim=-1)

            min_idxs = torch.argmin(dists, dim=1)
            tgt = tgt_pool[torch.arange(n_pairs, device=self.device), min_idxs]

        return src, tgt

    def _sample_proximity_pairs(
        self,
        src_candidates: torch.Tensor,
        tgt_candidates: torch.Tensor,
        n_pairs:        int,
        sigma:          float = 2.0,
        chunk_size:     int   = 100_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_pairs <= chunk_size:
            return self._sample_proximity_pairs_chunk(
                src_candidates, tgt_candidates, n_pairs, sigma=sigma)

        src_parts: list[torch.Tensor] = []
        tgt_parts: list[torch.Tensor] = []
        remaining = n_pairs
        while remaining > 0:
            cur = min(chunk_size, remaining)
            src_chunk, tgt_chunk = self._sample_proximity_pairs_chunk(
                src_candidates, tgt_candidates, cur, sigma=sigma)
            src_parts.append(src_chunk)
            tgt_parts.append(tgt_chunk)
            remaining -= cur
        return torch.cat(src_parts), torch.cat(tgt_parts)

    def _wire_new_neurons(self, old_limit: int, new_limit: int):
        """Grow axons and dendrites for newly activated neurons.

        New neurons form synapses with existing active neurons — both incoming
        (dendritic growth, ~1-2 weeks in biology) and outgoing (axonal growth,
        ~3-4 weeks). Initial weights are small (immature synapses, Bhatt 2009)
        and integrity is low so pruning removes them quickly if unused.
        """
        new_count  = new_limit - old_limit
        n_existing = old_limit
        n_new_syn  = max(1, int(new_count * n_existing * self._initial_density * 2))

        # Wire new neurons using proximity
        new_c = torch.arange(old_limit, new_limit, device=self.device)
        existing_c = torch.arange(old_limit, device=self.device)

        # Half incoming (existing -> new), half outgoing (new -> existing)
        half = n_new_syn // 2
        src_incoming, tgt_incoming = self._sample_proximity_pairs(existing_c, new_c, half, sigma=2.0)
        src_outgoing, tgt_outgoing = self._sample_proximity_pairs(new_c, existing_c, n_new_syn - half, sigma=2.0)

        src = torch.cat([src_incoming, src_outgoing])
        tgt = torch.cat([tgt_incoming, tgt_outgoing])
        no_self = src != tgt
        src, tgt = src[no_self], tgt[no_self]
        n_new_syn = src.shape[0]

        signs       = torch.where(self.is_inhibitory_tensor[src], -1.0, 1.0)
        new_weights = torch.rand(n_new_syn, device=self.device) * 0.02 * signs
        new_integ   = torch.full((n_new_syn,), 0.3, device=self.device)  # fragile

        self.indices          = torch.cat([self.indices,          torch.stack([src, tgt])], dim=1)
        self.weights_values   = torch.cat([self.weights_values,   new_weights])
        self.integrity_values = torch.cat([self.integrity_values, new_integ])
        self.eligibility      = torch.cat([self.eligibility,      torch.zeros(n_new_syn, device=self.device)])
        self._stp_u           = torch.cat([self._stp_u, torch.full((n_new_syn,), 0.3, device=self.device)])
        self._stp_x           = torch.cat([self._stp_x, torch.ones(n_new_syn,        device=self.device)])
        self._topology_changed = True

    def trigger_neurogenesis(self, surprise_level: float):
        if self.growth_cooldown > 0:
            self.growth_cooldown -= 1
            return
        self.epistemic_hunger += surprise_level
        if self.epistemic_hunger > self.hunger_threshold and self.active_limit < self._active_ceiling:
            old_limit          = self.active_limit
            self.active_limit  = min(self.active_limit + 5, self._active_ceiling)
            self.epistemic_hunger = 0.0
            self.growth_cooldown  = 300
            self._wire_new_neurons(old_limit, self.active_limit)

    # ------------------------------------------------------------------
    # STDP — biologically asymmetric (Bi & Poo 1998)
    # ------------------------------------------------------------------

    @property
    def critical_period_factor(self) -> float:
        """STDP rate multiplier from current brain age (Hensch 2005)."""
        import math
        return 1.0 + (self._cp_initial - 1.0) * math.exp(-self._age_ticks / self._cp_tau)

    def apply_stdp_learning(
        self,
        current_activity:          torch.Tensor,
        neuromodulator_multiplier: float,
        acetylcholine:             float = 0.5,
        dopamine:                  float = 0.5,
    ):
        # Critical-period scaling — younger brain learns faster (Hensch 2005)
        self._age_ticks += 1
        cp_scale = self.critical_period_factor
        neuromodulator_multiplier = neuromodulator_multiplier * cp_scale

        # Decay eligibility trace (τ ≈ 5 ticks)
        self.eligibility *= 0.8

        # Asymmetric trace decay: pre decays faster (narrower LTP window),
        # post decays slower (wider LTD window) — Bi & Poo 1998: τ- > τ+.
        self.pre_trace  *= 0.85   # T½ ≈ 4.3 ticks → τ+ ≈ 20 ms proxy
        self.post_trace *= 0.90   # T½ ≈ 6.6 ticks → τ- ≈ 40 ms proxy

        active_now_mask = current_activity > 0.5

        if not active_now_mask.any():
            # Still update traces even if nothing fires this tick
            return

        pre_idx  = self.indices[0]
        post_idx = self.indices[1]

        # Sign: inhibitory pre → negative direction for LTP and LTD
        signs = torch.where(self.is_inhibitory_tensor[pre_idx], -1.0, 1.0)

        # LTP/LTD use traces from BEFORE this tick's spikes are registered.
        # Traces reflect neurons that fired in PREVIOUS ticks only — correct
        # temporal ordering: pre must precede post (Bi & Poo 1998).

        # LTP: causal (pre fired before post fires now)
        # Magnitude A+ = 0.05, scales with pre trace (how recently pre fired)
        ltp_mask = (self.pre_trace[pre_idx] > 0.1) & active_now_mask[post_idx]
        if ltp_mask.any():
            # Scale by postsynaptic activity magnitude (calcium influx proxy)
            # ACh broadens STDP window — higher ACh → more LTP (Hasselmo 2003)
            post_act_mag = current_activity[post_idx][ltp_mask]
            ach_boost = 1.0 + acetylcholine * 0.3
            delta_ltp = (0.05
                         * self.pre_trace[pre_idx][ltp_mask]
                         * post_act_mag
                         * neuromodulator_multiplier
                         * ach_boost)
            self.eligibility[ltp_mask] += delta_ltp * signs[ltp_mask]
            self.integrity_values[ltp_mask] = torch.clamp(
                self.integrity_values[ltp_mask] + 0.1 * dopamine, 0.0, 2.0)

        # LTD: anti-causal (post fired before pre fires now)
        # Magnitude A- = 0.015  (< A+, asymmetry from Bi & Poo 1998)
        ltd_mask = active_now_mask[pre_idx] & (self.post_trace[post_idx] > 0.1)
        if ltd_mask.any():
            delta_ltd = (0.015
                         * self.post_trace[post_idx][ltd_mask]
                         * neuromodulator_multiplier)
            self.eligibility[ltd_mask] -= delta_ltd * signs[ltd_mask]

        # Passive synaptic integrity decay (pruning of unused connections, τ ≈ 10000 ticks)
        self.integrity_values = torch.clamp(self.integrity_values - 0.0001, min=0.0)

        # Consolidate eligibility trace into weights, gated/modulated by dopamine
        # High dopamine -> consolidate LTP/LTD; low dopamine -> suppress/reduce consolidation
        new_weights = self.weights_values + self.eligibility * dopamine * 0.1

        # Enforce Dale's Principle: prevent sign flipping (clamping)
        # Inhibitory synapses (pre-synaptic is inhibitory) must stay in [-1.0, 0.0]
        # Excitatory synapses (pre-synaptic is excitatory) must stay in [0.0, 1.0]
        is_inh_syn = self.is_inhibitory_tensor[self.indices[0]]
        self.weights_values = torch.where(
            is_inh_syn,
            torch.clamp(new_weights, -1.0, 0.0),
            torch.clamp(new_weights, 0.0, 1.0)
        )

        # NOW register spikes into traces — after LTP/LTD, so this tick's
        # activity only influences plasticity from the NEXT tick onward.
        self.pre_trace[active_now_mask]  = 1.0
        self.post_trace[active_now_mask] = 1.0

    # ------------------------------------------------------------------
    # Cortical lateral inhibition — GABAergic basket cells (Buzsáki 2004)
    # ------------------------------------------------------------------

    def apply_lateral_inhibition(
        self,
        activity:      torch.Tensor,
        target_sparsity: float = 0.05,
        acetylcholine:   float = 0.5,
    ) -> torch.Tensor:
        """Cell-autonomous lateral inhibition.

        Sparsity and lateral competition are maintained through dynamic homeostatic
        thresholds and population somatic feedback during integration, removing the
        need for global top-k sorting.
        """
        return activity

    # ------------------------------------------------------------------
    # Structural plasticity
    # ------------------------------------------------------------------

    def perform_sleep_pruning(self):
        """Prune weak/unused synapses during sleep (Tononi 2006 SHY).

        Synapses with low integrity (integrity < 0.05) and very small weights
        are selectively pruned.
        """
        weak_mask = (self.integrity_values < 0.05) & (self.weights_values.abs() < 0.01)
        alive_mask = ~weak_mask

        if not alive_mask.all():
            self.indices          = self.indices[:, alive_mask]
            self.weights_values   = self.weights_values[alive_mask]
            self.integrity_values = self.integrity_values[alive_mask]
            self.eligibility      = self.eligibility[alive_mask]
            self._stp_u           = self._stp_u[alive_mask]
            self._stp_x           = self._stp_x[alive_mask]
            self._topology_changed = True

    def synaptogenesis_and_pruning(self, active_nodes: torch.Tensor, energy_level: float):
        # Wake phase only handles growth/synaptogenesis, pruning is sleep-driven.

        # Growth: spawn new synapses between co-active neurons
        if energy_level > 1000.0 and random.random() < 0.05:
            active_idx = torch.where(active_nodes[:self.active_limit] > 0.5)[0]
            if len(active_idx) > 1:
                n_new = 20
                src, tgt = self._sample_proximity_pairs(active_idx, active_idx, n_new, sigma=1.5)
                no_self = src != tgt
                src, tgt = src[no_self], tgt[no_self]
                if src.shape[0] > 0:
                    k = src.shape[0]
                    signs         = torch.where(self.is_inhibitory_tensor[src], -1.0, 1.0)
                    self.indices          = torch.cat([self.indices, torch.stack([src, tgt])], dim=1)
                    self.weights_values   = torch.cat([self.weights_values,
                                                       torch.rand(k, device=self.device) * 0.1 * signs])
                    self.integrity_values = torch.cat([self.integrity_values,
                                                       torch.ones(k, device=self.device) * 0.5])
                    self.eligibility      = torch.cat([self.eligibility,
                                                       torch.zeros(k, device=self.device)])
                    self._stp_u           = torch.cat([self._stp_u,
                                                       torch.full((k,), 0.3, device=self.device)])
                    self._stp_x           = torch.cat([self._stp_x,
                                                       torch.ones(k, device=self.device)])
                    self._topology_changed = True

    # ------------------------------------------------------------------
    # Synaptic homeostasis (Turrigiano 1998)
    # ------------------------------------------------------------------

    def maintain_homeostasis(self):
        """Normalise incoming weight sums to prevent runaway excitation/silence.

        Uses scatter_add on edge indices instead of sparse.sum().to_dense() —
        O(edges) with no full-matrix materialisation (Turrigiano 1998).
        """
        post_idx = self.indices[1]
        # Sum |w| per post-synaptic neuron via scatter — touches only existing edges
        incoming_sum = torch.zeros(self.num_nodes, device=self.device)
        incoming_sum.scatter_add_(0, post_idx, self.weights_values.abs())
        target = self._homeostatic_target

        overloaded  = incoming_sum > target
        underloaded = (incoming_sum > 0) & (incoming_sum < target * 0.1)

        scale_factors = torch.ones(self.num_nodes, device=self.device)
        if overloaded.any():
            scale_factors[overloaded] = target / incoming_sum[overloaded]
        if underloaded.any():
            scale_factors[underloaded] = torch.clamp(
                (target * 0.5) / incoming_sum[underloaded], 1.0, 2.0)

        self.weights_values *= scale_factors[post_idx]

    # ------------------------------------------------------------------
    # Cortisol-mediated glutamate excitotoxicity (Cerqueira 2007)
    # ------------------------------------------------------------------

    def apply_cortisol_damage(self, cortisol_level: float):
        """Chronic cortisol → dendritic atrophy in prefrontal and hippocampal neurons.

        Mechanism: sustained HPA activation → glutamate excitotoxicity →
        Ca²⁺ overload → dendritic spine retraction (McEwen 2007).
        Selects most active post-synaptic neurons (highest activation =
        highest Ca²⁺ load) rather than random victim.
        """
        if cortisol_level <= 0.5 or random.random() >= 0.1:
            return
        damage_chance = (cortisol_level - 0.5) * 0.05
        if random.random() >= damage_chance:
            return

        # Pick a vulnerable neuron (most active incoming weights = most Ca²⁺)
        incoming_sum = torch.zeros(self.num_nodes, device=self.device)
        incoming_sum.scatter_add_(0, self.indices[1], self.weights_values.abs())
        # Avoid zeros (unconnected neurons)
        candidates   = torch.where(incoming_sum > 0)[0]
        if len(candidates) == 0:
            return
        victim = int(candidates[torch.argmax(incoming_sum[candidates])])

        victim_mask = self.indices[1] == victim   # afferents onto victim
        if victim_mask.any():
            self.weights_values[victim_mask]   *= 0.8
            self.integrity_values[victim_mask]  = torch.clamp(
                self.integrity_values[victim_mask] - 0.2, min=0.0)
            print(f'    [ПСИХИАТРИЯ] Глутаматная эксайтотоксичность! '
                  f'Кортизол повредил дендриты нейрона {victim}.')
