"""CrossModalBinder — Hebbian binding between language and sensory modalities.

Biological basis (Damasio 1989 — convergence zones):
  Antonio Damasio's convergence zone theory posits that the brain does NOT
  store unified "concept" representations.  Instead, each modality stores
  its OWN representation, and multimodal concepts emerge from *convergence
  zones* — cortical regions that, when re-activated, trigger co-activation
  of the relevant modality-specific regions via back-projections.

  Example: The concept "apple" is not stored as a single node.  Instead:
    * Visual cortex stores colour/shape (red, round)
    * Auditory cortex stores the word phonology (/ˈæp.əl/)
    * Somatosensory cortex stores the tactile texture (smooth, waxy)
    * Gustatory cortex stores the taste (sweet, tart)
    * LanguageCortex.Wernicke stores the semantic binding
    * EntorhinalGrid holds the position in conceptual space

  The convergence zone (in inferior temporal cortex, angular gyrus) fires
  when any subset of these patterns is present, triggering retrieval of the
  rest via back-projections.  This is cross-modal Hebbian binding.

Why this solves the embodied grounding problem:
  Without this binder, "apple" is just a token ID.  With it, hearing the
  word /ˈæp.əl/ activates the visual cortex's apple-shape representation,
  the semantic vector in Wernicke, and the entorhinal position — giving the
  word grounded, multimodal meaning.

Implementation:
  * A weight matrix W_bind[i][j] records how strongly modality i predicts
    modality j.
  * When modalities co-activate within the same tick, Hebbian update:
    ΔW_bind[i][j] += η × (act_i > threshold) × (act_j > threshold)
  * At recall: given a partial activation of modality i, the binder
    computes a recall_signal for modality j via W_bind[i][j].
  * All cross-modal pairs: text↔audio, text↔vision, audio↔vision,
    text↔interoception, etc.
"""

from __future__ import annotations

import numpy as np


# Threshold for counting a modality as "active" at this tick
_ACT_THRESHOLD = 0.1

# Maximum weight magnitude (prevents Hebbian runaway)
_W_MAX = 1.0

# Hebbian learning rate (deliberately slow — binding takes experience)
_ETA = 0.002


class CrossModalBinder:
    """Convergence-zone cross-modal Hebbian binder.

    Maintains a weight matrix between every pair of registered modality
    embeddings.  When multiple modalities are simultaneously active,
    Hebbian co-activation strengthens their mutual bindings.

    Parameters
    ----------
    modality_dims : dict[str, int]
        Mapping of modality name → embedding dimensionality.
        E.g. {'text': 256, 'audio': 64, 'vision': 512}.
    """

    def __init__(self, modality_dims: dict[str, int]) -> None:
        self.modality_dims = dict(modality_dims)
        self._modalities   = list(modality_dims.keys())
        n = len(self._modalities)

        # Weight matrices: W_bind[i][j] maps modality_i → modality_j
        # Shape: (dim_j, dim_i) — matrix-multiply to project
        rng = np.random.default_rng(99)
        self._W: dict[tuple[int, int], np.ndarray] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dim_i = self._modality_dims_list[i]
                dim_j = self._modality_dims_list[j]
                self._W[(i, j)] = rng.standard_normal(
                    (dim_j, dim_i)).astype(np.float32) * 0.001

        # Binding strength per modality pair (scalar summary for Brain)
        self.binding_strengths: dict[str, float] = {}

    # ------------------------------------------------------------------

    @property
    def _modality_dims_list(self) -> list[int]:
        return [self.modality_dims[m] for m in self._modalities]

    def _mod_idx(self, name: str) -> int:
        return self._modalities.index(name)

    # ------------------------------------------------------------------

    def update(self, activations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Hebbian update and cross-modal recall.

        Parameters
        ----------
        activations : dict[str, np.ndarray]
            Current embedding for each modality.  Missing modalities are
            treated as absent (zero vector; no Hebbian update for that pair).

        Returns
        -------
        recall_signals : dict[str, np.ndarray]
            For each modality, a recall vector reconstructed from ALL other
            currently active modalities.  These can be additively blended
            into the modality's own processing stream.
        """
        # Normalise / fit all provided activations
        fitted: dict[str, np.ndarray] = {}
        for name, act in activations.items():
            if name not in self.modality_dims:
                continue
            dim = self.modality_dims[name]
            a   = np.asarray(act, dtype=np.float32).ravel()
            fitted[name] = _fit(a, dim)

        active_mods = list(fitted.keys())
        n_active    = len(active_mods)

        # ----------------------------------------------------------------
        # 1. Hebbian update for every co-active modality pair
        # ----------------------------------------------------------------
        for i_name in active_mods:
            for j_name in active_mods:
                if i_name == j_name:
                    continue
                i = self._mod_idx(i_name)
                j = self._mod_idx(j_name)
                a_i = fitted[i_name]
                a_j = fitted[j_name]

                # Only update when BOTH are "active" (above threshold)
                i_active = float(np.mean(np.abs(a_i))) > _ACT_THRESHOLD
                j_active = float(np.mean(np.abs(a_j))) > _ACT_THRESHOLD
                if not (i_active and j_active):
                    continue

                # Outer-product Hebbian update
                dW = _ETA * np.outer(a_j, a_i)   # (dim_j, dim_i)
                self._W[(i, j)] = np.clip(self._W[(i, j)] + dW, -_W_MAX, _W_MAX)

                # Track binding strength (Frobenius norm, normalised)
                key = f'{i_name}↔{j_name}'
                self.binding_strengths[key] = float(
                    np.linalg.norm(self._W[(i, j)]) /
                    (self._W[(i, j)].size ** 0.5 + 1e-8))

        # ----------------------------------------------------------------
        # 2. Cross-modal recall for each target modality
        # ----------------------------------------------------------------
        recall_signals: dict[str, np.ndarray] = {}
        for j_name in self.modality_dims:
            j      = self._mod_idx(j_name)
            dim_j  = self.modality_dims[j_name]
            recall = np.zeros(dim_j, dtype=np.float32)
            n_contributors = 0

            for i_name in active_mods:
                if i_name == j_name:
                    continue
                i   = self._mod_idx(i_name)
                a_i = fitted[i_name]
                if float(np.mean(np.abs(a_i))) <= _ACT_THRESHOLD:
                    continue
                # Project modality_i activation through binding weight
                contrib = self._W[(i, j)] @ a_i   # (dim_j,)
                recall  += contrib
                n_contributors += 1

            if n_contributors > 0:
                recall /= n_contributors  # average across contributing modalities
            recall_signals[j_name] = recall

        return recall_signals

    # ------------------------------------------------------------------

    def strongest_binding(self) -> str | None:
        """Return the modality pair with the highest binding strength."""
        if not self.binding_strengths:
            return None
        return max(self.binding_strengths, key=self.binding_strengths.get)

    def get_binding_strength(self, mod_a: str, mod_b: str) -> float:
        """Return scalar binding strength between two modalities."""
        key = f'{mod_a}↔{mod_b}'
        return self.binding_strengths.get(key, 0.0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate 1-D float32 array to length n."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))
