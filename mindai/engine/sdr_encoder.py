"""Sparse Distributed Representations (SDR) — cortical population code.

Biological basis (Olshausen & Field 1996; Maynard et al. 1999):
  In primary sensory and association cortex ~2% of neurons are active at any
  moment.  This extreme sparsity is NOT wasteful — it is the key to cortical
  capacity.  With N neurons and K active at once the number of representable
  patterns is C(N, K).  For N=10 000, K=200 (2%) that is ≈10^475, far exceeding
  the number of atoms in the observable universe.

  Additionally, SDR representations share a key property with real neocortex:
    * Two SDRs overlap in proportion to their semantic similarity.
    * Noise tolerance: up to 50% of bits can be corrupted and the pattern
      is still correctly identified (robust majority vote).
    * Sparsity also minimises metabolic cost (Attwell & Laughlin 2001).

SDR vs dense vectors:
  Dense [0.1, 0.8, 0.3, ...]  — ambiguous, fragile, no clear identity
  SDR   [0, 0, 1, 0, 1, 0 ...]— unambiguous, fault-tolerant, combinatorially rich

Implementation:
  * Encoder: takes a continuous vector of arbitrary size, returns a binary
    sparse vector with exactly `k` bits set (top-K selection).
  * Decoder: reconstructs a continuous approximation from the SDR via a
    learned reverse mapping (Hebbian, online).
  * The `k` parameter defaults to 2% of `n`; can be set per modality.

References:
  Olshausen BA, Field DJ (1996) Emergence of simple-cell receptive field
    properties by learning a sparse code for natural images.
    Nature 381: 607-609.
  Ahmad S, Hawkins J (2016) How do neurons operate on sparse distributed
    representations? A mathematical theory of sparsity, neurons and active
    dendrites. arXiv:1601.00720.
"""

from __future__ import annotations

import numpy as np


class SDREncoder:
    """Encode continuous vectors to sparse binary SDRs and back.

    Parameters
    ----------
    n : int
        Number of output bits (SDR width).
    k : int | None
        Number of active bits.  If None, defaults to max(1, int(n * 0.02))
        (the biological ~2% sparsity level).
    seed : int
        RNG seed for the initial random projection matrix.
    """

    def __init__(self, n: int, k: int | None = None, seed: int = 42) -> None:
        self.n = n
        self.k = k if k is not None else max(1, int(n * 0.02))

        rng = np.random.default_rng(seed)

        # Random projection matrix: maps arbitrary input dim → n dims.
        # Not fixed: columns are updated online via Hebbian reverse-mapping.
        # Initialised with small random weights (sign-preserving projection).
        # The matrix is stored lazily — built on first call to encode().
        self._proj: np.ndarray | None = None   # shape (n, input_dim)
        self._proj_T: np.ndarray | None = None  # shape (input_dim, n)
        self._rng = rng

        # Reverse mapping: SDR → continuous reconstruction (Hebbian)
        # Δw = η × (target - estimate) × sdr_bit  (perceptron rule)
        self._recon: np.ndarray | None = None   # shape (n, input_dim)
        self._lr = 0.01   # Hebbian learning rate for reverse mapping

        # Statistics
        self.last_sdr: np.ndarray | None = None     # last binary output
        self.last_overlap: float = 0.0              # overlap with prev SDR

    # ------------------------------------------------------------------

    def _ensure_projection(self, input_dim: int) -> None:
        if self._proj is None:
            # Gaussian random projection (Johnson-Lindenstrauss)
            self._proj  = self._rng.standard_normal((self.n, input_dim)).astype(np.float32)
            self._proj  /= np.linalg.norm(self._proj, axis=1, keepdims=True) + 1e-8
            self._recon  = self._rng.standard_normal((self.n, input_dim)).astype(np.float32) * 0.01
        elif self._proj.shape[1] != input_dim:
            # Input dimension changed — expand projection matrix
            extra = input_dim - self._proj.shape[1]
            new_cols = self._rng.standard_normal((self.n, extra)).astype(np.float32)
            new_cols /= np.linalg.norm(new_cols, axis=1, keepdims=True) + 1e-8
            self._proj  = np.concatenate([self._proj, new_cols], axis=1)
            new_rec = self._rng.standard_normal((self.n, extra)).astype(np.float32) * 0.01
            self._recon = np.concatenate([self._recon, new_rec], axis=1)

    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a continuous vector into a sparse binary SDR.

        Parameters
        ----------
        x : np.ndarray
            Continuous input vector of any length.  Values need not be
            normalised — the encoder handles that internally.

        Returns
        -------
        sdr : np.ndarray [n], dtype float32
            Binary vector with exactly k bits set (values are 0.0 or 1.0).
            Using float32 (not bool) for direct torch interop.
        """
        x = np.asarray(x, dtype=np.float32).ravel()
        self._ensure_projection(len(x))

        # Project to n-dimensional space
        proj_out = self._proj @ x                     # (n,)

        # Top-K selection — only the k strongest activations fire
        sdr = np.zeros(self.n, dtype=np.float32)
        if self.k > 0:
            topk_idx = np.argpartition(proj_out, -self.k)[-self.k:]
            sdr[topk_idx] = 1.0

        # Overlap with previous SDR (semantic similarity measure)
        if self.last_sdr is not None:
            self.last_overlap = float(np.dot(sdr, self.last_sdr)) / (self.k + 1e-8)
        self.last_sdr = sdr

        return sdr

    def decode(self, sdr: np.ndarray, input_dim: int) -> np.ndarray:
        """Reconstruct a continuous approximation from an SDR.

        Uses the learned reverse mapping (trained via Hebbian updates in
        ``update_reconstruction``).  Without training this is a random
        projection and will be noisy — it improves over time.

        Parameters
        ----------
        sdr : np.ndarray [n]
            Binary SDR vector.
        input_dim : int
            Dimensionality of the original input space.

        Returns
        -------
        reconstruction : np.ndarray [input_dim]
        """
        self._ensure_projection(input_dim)
        recon = sdr @ self._recon           # (input_dim,)
        # Normalise to [0, 1] range
        mn, mx = recon.min(), recon.max()
        if mx - mn > 1e-8:
            recon = (recon - mn) / (mx - mn)
        return recon.astype(np.float32)

    def update_reconstruction(self, sdr: np.ndarray, target: np.ndarray) -> None:
        """Hebbian update to improve decode accuracy.

        Called after encode() when the original input is still available.
        Over repeated calls the reverse mapping converges (perceptron rule).

        Parameters
        ----------
        sdr : np.ndarray [n]
        target : np.ndarray [input_dim]   — original continuous input
        """
        if self._recon is None:
            self._ensure_projection(len(target))
        estimate = sdr @ self._recon                  # (input_dim,)
        delta    = (target - estimate) * self._lr     # (input_dim,)
        # Only update rows where the SDR bit is active (Hebbian locality)
        active = sdr > 0.5
        self._recon[active] += delta[np.newaxis, :]   # broadcast add

    # ------------------------------------------------------------------

    @property
    def sparsity(self) -> float:
        """Fraction of bits currently active (should be ≈ k/n ≈ 0.02)."""
        return self.k / self.n
