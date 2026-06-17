"""Entorhinal cortex — grid cells + place cell input to hippocampus.

Biological basis
----------------
Medial entorhinal cortex (MEC) layer II contains GRID CELLS that fire in
a hexagonal lattice pattern across the animal's environment (Hafting 2005).
Different MEC modules have different grid spacings (1.4× scaling between
adjacent modules) and orientations — together they form a multi-scale
positional code.

PLACE CELLS in the hippocampus (CA1/CA3) emerge from grid cell input:
each place cell's firing field is a unique combination of grid cell
phases that peaks at one specific location (O'Keefe & Dostrovsky 1971).

Why MindAI needs this
---------------------
For a text/multimodal agent there is no physical "location" — but there
IS a NARRATIVE/SEMANTIC position that benefits from the same encoding:

    - What is the current "scene" the brain is processing?
    - When discussing a story, which "location" in the narrative are we?
    - Concept neighbourhoods in semantic space: "tree" near "forest"

The grid encoding gives a continuous, low-dimensional manifold position
that the hippocampus can use to generate place-like cells responsive to
clusters of related concepts. This emerges from STDP without any explicit
spatial input — the brain treats topic transitions like spatial movement.

Function
--------
- track_position(token_or_concept_id) — moves through latent semantic space
  (delta proportional to embedding distance to previous concept)
- get_grid_activity() — returns multi-scale hexagonal activations
- These feed into hippocampus as the EC layer II input.

References
----------
- O'Keefe J, Dostrovsky J (1971). The hippocampus as a spatial map.
  Brain Res 34: 171-175.
- Hafting T et al. (2005). Microstructure of a spatial map in the
  entorhinal cortex. Nature 436: 801-806.
- Stensola H et al. (2012). The entorhinal grid map is discretized.
  Nature 492: 72-78. (4 modules, 1.4× scaling)
"""

from __future__ import annotations

import numpy as np


class EntorhinalGrid:
    """Multi-scale hexagonal grid cells over a latent 2D semantic position."""

    # Stensola 2012: 4 discrete grid modules with 1.4× spacing scaling
    _MODULE_SCALES = (1.0, 1.4, 1.96, 2.74)

    def __init__(
        self,
        cells_per_module: int = 64,
        embed_dim:        int = 2,    # 2D semantic position (manifold)
        rng_seed:         int = 7,
    ):
        rng = np.random.default_rng(rng_seed)
        self.cells_per_module = cells_per_module
        self.num_modules      = len(self._MODULE_SCALES)
        # Per-cell phase offsets (random phase within the module)
        self._phase_offsets = [
            rng.random(cells_per_module) * 2 * np.pi
            for _ in self._MODULE_SCALES
        ]
        # Three hex basis vectors (60° apart) for true hexagonal periodicity
        self._hex_basis = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
            [-0.5, np.sqrt(3) / 2],
        ], dtype=np.float32)

        # Latent semantic position in [-1, 1]^embed_dim
        self.position = np.zeros(embed_dim, dtype=np.float32)
        self._embed_dim = embed_dim

        # Concept embeddings for advance() — populated by callers
        self._concept_pos: dict[int, np.ndarray] = {}
        self._rng = rng

    # ------------------------------------------------------------------

    @property
    def total_cells(self) -> int:
        return self.num_modules * self.cells_per_module

    def _grid_activation(self, position: np.ndarray, scale: float,
                         phase_offset: np.ndarray) -> np.ndarray:
        """One module's hexagonal grid response at given position."""
        # Project position onto each hex basis vector, scaled
        scaled = position[:2] / scale   # use first 2 dims as 2D map
        # Hex grid = sum of cos along three 60°-separated axes
        grid = np.zeros(self.cells_per_module, dtype=np.float32)
        for axis in self._hex_basis:
            grid += np.cos(2 * np.pi * np.dot(scaled, axis)
                           + phase_offset)
        # Normalise: cosine sum range [-3, 3] → [0, 1]
        grid = (grid + 3.0) / 6.0
        return grid

    # ------------------------------------------------------------------

    def get_grid_activity(self) -> np.ndarray:
        """Concatenated grid activations across all modules."""
        outs = [
            self._grid_activation(self.position, scale, self._phase_offsets[m])
            for m, scale in enumerate(self._MODULE_SCALES)
        ]
        return np.concatenate(outs).astype(np.float32)

    def advance(self, concept_id: int, jump_size: float = 0.05) -> None:
        """Move semantic position based on incoming concept.

        Each unique concept_id is assigned a fixed random direction the
        first time it's seen. Repeated occurrences nudge the position
        toward that concept's "place" in the manifold. Topic shifts
        (different concept ids) cause grid cells to fire differently —
        downstream hippocampal place cells will encode topic transitions.
        """
        if concept_id not in self._concept_pos:
            # New concept: assign a random anchor in [-1, 1]^embed_dim
            self._concept_pos[concept_id] = (
                self._rng.random(self._embed_dim).astype(np.float32) * 2.0 - 1.0)
        target = self._concept_pos[concept_id]
        # Move smoothly toward concept anchor
        delta = (target - self.position) * jump_size
        self.position = np.clip(self.position + delta, -1.0, 1.0)

    def reset(self) -> None:
        self.position[:] = 0.0
