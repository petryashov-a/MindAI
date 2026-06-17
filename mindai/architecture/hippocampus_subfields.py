"""Hippocampal subfields — DG / CA3 / CA1 with pattern separation and completion.

Biological basis
----------------
The hippocampus is not a uniform structure. It has three main subfields,
each with a distinct computational role (Marr 1971, McNaughton & Morris
1987, Treves & Rolls 1994):

    Dentate Gyrus (DG)
      - Sparse, expanded representation (~5× more granule cells than EC inputs)
      - Strong inhibition (basket cells) → very few active per pattern (~1-2%)
      - Function: PATTERN SEPARATION — orthogonalises similar inputs so
        episodes don't overwrite each other. Similar inputs map to
        well-separated DG patterns. (Leutgeb 2007)
      - Adult neurogenesis specifically here (Eriksson 1998).

    CA3
      - Recurrent collateral network (one neuron contacts ~12000 others)
      - Auto-associative memory: complete a stored pattern from a partial
        cue. (Marr 1971; Hopfield-style attractor)
      - Receives DG input via mossy fibres (very strong, sparse) AND
        direct EC input (weak, distributed).

    CA1
      - Output stage. Compares CA3 reconstruction with current EC input.
      - Mismatch → novelty signal sent to dopamine system (Lisman & Grace
        2005 — "VTA-hippocampal loop").
      - Sole projection from hippocampus back to cortex via subiculum.

Function in MindAI
------------------
Replaces the flat episodic buffer with a three-stage pipeline:

    [activity from cortex]
        ↓ EC layer III (direct path, weak)
        ↓ EC layer II (DG path)
    DG (sparse 1% activity) — pattern separator
        ↓ mossy fibres
    CA3 — auto-associative recall (returns stored pattern given cue)
        ↓ Schaffer collaterals
    CA1 — comparator (CA3 recall vs current input)
        ↓
    output → cortex (replay) + novelty signal → DA

This solves catastrophic forgetting: similar new episodes get separated
in DG before being stored, so they don't overwrite older similar memories.
Recall is content-addressable: a partial cue retrieves the full pattern.

References
----------
- Marr D (1971). Simple memory: a theory for archicortex.
- Treves A, Rolls ET (1994). Computational analysis of the role of the
  hippocampus in memory. Hippocampus 4: 374-391.
- Leutgeb JK et al. (2007). Pattern separation in the dentate gyrus and
  CA3 of the hippocampus. Science 315: 961-966.
- Lisman J, Grace AA (2005). The hippocampal-VTA loop: controlling the
  entry of information into long-term memory. Neuron 46: 703-713.
"""

from __future__ import annotations

import numpy as np


class HippocampalSubfields:
    """DG → CA3 → CA1 pipeline with pattern separation and completion."""

    def __init__(
        self,
        input_size:    int   = 1024,
        dg_size:       int   = 4096,   # 4× expansion (DG > EC)
        ca3_size:      int   = 1024,
        ca1_size:      int   = 1024,
        dg_sparsity:   float = 0.01,   # ~1% active in DG (very sparse)
        ca3_sparsity:  float = 0.05,   # ~5% in CA3
        rng_seed:      int   = 42,
    ):
        self.input_size  = input_size
        self.dg_size     = dg_size
        self.ca3_size    = ca3_size
        self.ca1_size    = ca1_size
        self.dg_sparsity = dg_sparsity
        self.ca3_sparsity = ca3_sparsity

        rng = np.random.default_rng(rng_seed)
        # EC → DG: random sparse projection (~3% non-zero)
        # Sparse + random gives orthogonalising property (Johnson-Lindenstrauss)
        self._W_ec_dg = (rng.random((input_size, dg_size)) < 0.03).astype(np.float32)
        # DG → CA3: mossy fibres are very sparse but strong (one DG → ~14 CA3)
        self._W_dg_ca3 = np.zeros((dg_size, ca3_size), dtype=np.float32)
        for d in range(dg_size):
            targets = rng.choice(ca3_size, 14, replace=False)
            self._W_dg_ca3[d, targets] = rng.random(14) * 0.5 + 0.5
        # CA3 → CA1: Schaffer collaterals
        self._W_ca3_ca1 = (rng.random((ca3_size, ca1_size)) < 0.10).astype(np.float32) * 0.5
        # CA3 recurrent (auto-associative)
        self._W_ca3_rec = np.zeros((ca3_size, ca3_size), dtype=np.float32)
        self._has_stored = False

        # CA3 stored patterns — for retrieval comparison
        self._stored_patterns: list[np.ndarray] = []
        # Episode metadata aligned with stored patterns
        self._episode_meta: list[dict] = []

        # Last DG activation — exposed for inspection
        self.last_dg:  np.ndarray | None = None
        self.last_ca3: np.ndarray | None = None
        self.last_ca1: np.ndarray | None = None
        # Novelty signal (CA1 mismatch) for DA modulation (Lisman & Grace 2005)
        self.novelty_signal: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kwta(x: np.ndarray, sparsity: float) -> np.ndarray:
        """k-Winners-Take-All — top fraction stays, rest zeroed."""
        n = x.size
        k = max(1, int(n * sparsity))
        if k >= n:
            return x.copy()
        thr = np.partition(x, n - k)[n - k]
        out = np.where(x >= thr, x, 0.0).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Forward pass — encode an incoming pattern through DG→CA3→CA1
    # ------------------------------------------------------------------

    def encode(self, ec_input: np.ndarray) -> dict:
        """Run input through full DG→CA3→CA1 pipeline.

        Returns a dict with all subfield activations and the novelty signal.
        """
        ec = ec_input[:self.input_size].astype(np.float32)
        if ec.size < self.input_size:
            ec = np.pad(ec, (0, self.input_size - ec.size))

        # --- DG: sparse expansion, pattern separation ---
        dg_raw = ec @ self._W_ec_dg
        dg     = self._kwta(dg_raw, self.dg_sparsity)
        self.last_dg = dg

        # --- CA3: mossy fibre input + recurrent retrieval ---
        ca3_input = dg @ self._W_dg_ca3
        # Recurrent step (one pass, attractor approximation)
        if self._has_stored:
            ca3_input = ca3_input + ca3_input @ self._W_ca3_rec * 0.3
        ca3 = self._kwta(ca3_input, self.ca3_sparsity)
        self.last_ca3 = ca3

        # --- CA1: read-out + comparison ---
        ca1 = ca3 @ self._W_ca3_ca1
        ca1 = self._kwta(ca1, self.ca3_sparsity)
        self.last_ca1 = ca1

        # --- Novelty: mismatch between CA3 recall and stored patterns ---
        # If CA3 looks like nothing we've stored before → high novelty
        if self._stored_patterns:
            sims = [float(np.dot(ca3, p) / (np.linalg.norm(ca3) * np.linalg.norm(p) + 1e-9))
                    for p in self._stored_patterns[-50:]]   # last 50 only
            best = max(sims) if sims else 0.0
            self.novelty_signal = max(0.0, 1.0 - best)
        else:
            self.novelty_signal = 1.0

        return {
            'dg':       dg,
            'ca3':      ca3,
            'ca1':      ca1,
            'novelty':  self.novelty_signal,
        }

    # ------------------------------------------------------------------
    # Storage & recall
    # ------------------------------------------------------------------

    def store(self, episode: dict) -> None:
        """Add a CA3 pattern to the auto-associative store.

        Strengthens recurrent CA3 weights for co-active units (Hebbian
        outer product) — classic Hopfield encoding.
        """
        if self.last_ca3 is None:
            return
        ca3 = self.last_ca3
        active = ca3 > 0
        if active.sum() < 2:
            return
        # Hebbian outer product on active units only — sparse update
        idx = np.where(active)[0]
        self._W_ca3_rec[np.ix_(idx, idx)] += np.outer(ca3[idx], ca3[idx]) * 0.05
        np.fill_diagonal(self._W_ca3_rec, 0.0)
        # Cap to prevent runaway
        np.clip(self._W_ca3_rec, -1.0, 1.0, out=self._W_ca3_rec)
        self._has_stored = True
        self._stored_patterns.append(ca3.copy())
        self._episode_meta.append(episode)
        # Bound memory
        if len(self._stored_patterns) > 1000:
            self._stored_patterns = self._stored_patterns[-1000:]
            self._episode_meta    = self._episode_meta[-1000:]

    def recall(self, partial_cue: np.ndarray) -> np.ndarray | None:
        """Pattern completion: given a noisy/partial CA3-like cue,
        return the closest stored pattern.

        This is the core hippocampal contribution: episodic recall from
        an associative cue (e.g., a sound triggering a full memory).
        """
        if not self._stored_patterns:
            return None
        cue   = partial_cue[:self.ca3_size]
        norms = [np.linalg.norm(p) for p in self._stored_patterns]
        sims  = [float(np.dot(cue, p) / (np.linalg.norm(cue) * n + 1e-9))
                 for p, n in zip(self._stored_patterns, norms)]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] < 0.2:
            return None
        return self._stored_patterns[best_idx].copy()
