"""Biological motion detector — pSTS analog.

Biological basis (Oram & Perrett 1994; Grossman & Blake 2002):
  pSTS (posterior superior temporal sulcus) responds selectively to biological
  motion (e.g. point-light displays of walking/jumping), receiving motion
  signals from Area MT+/V5 (Gross & Blake 1999).
  Crucially, pSTS fires strongly to coherent, hierarchically structured
  movement with an internal source, whereas random motion or global uniform
  motion (camera pan) is ignored.

  Three hallmarks of biological motion that pSTS detects:
    1. Temporal autocorrelation: biological motion has smooth, predictable
       trajectories (muscles can't make discontinuous jumps). Pearson r correlation
       between consecutive frames is high.
    2. Structured variance: different parts of the visual field move at
       different rates simultaneously (limbs have phase offsets).
    3. Periodicity: walking, reaching, chopping — periodic rhythms in the 0.5–4 Hz range.

Implementation:
  We maintain a short history of the visual input (5 ticks ≈ 500 ms).
  Each tick we compute:
    - temporal_autocorr: Pearson r between t and t-1 frames (smooth = biological)
    - structured_variance: std of per-pixel variance (uniform motion → near 0)
    - periodicity: spectral power in 1–4 Hz band via a lightweight IIR bandpass

  Final score is a weighted product in [0, 1]. Score > threshold → biological
  motion detected. The score is returned as a float used to:
    1. Modulate STDP rate on mirror_neurons (via brain.py)
    2. Suppress the mirror signal when score is low (background noise, static)

  Operates entirely on CPU (numpy) — the visual buffer is already a numpy slice.
  Computational cost: O(vision_size) per tick.
"""

from __future__ import annotations
import numpy as np
from collections import deque


_HISTORY = 6          # ticks of visual history
_BP_A    = 0.75       # IIR bandpass coefficient (≈ 1–4 Hz passband at 10 Hz)


class BiologicalMotionDetector:
    """pSTS analog: scores how 'biological' the current visual input looks."""

    def __init__(self, vision_size: int):
        self.vision_size = vision_size
        self._history: deque[np.ndarray] = deque(maxlen=_HISTORY)
        # IIR bandpass state for periodicity detection
        self._bp_state = np.zeros(vision_size, dtype=np.float32)
        self._score    = 0.0   # last computed score, cached for brain.py

    @property
    def score(self) -> float:
        return self._score

    def update(self, visual_frame: np.ndarray) -> float:
        """Feed one frame; return biological-motion score in [0, 1].

        visual_frame: numpy float32 array of shape (vision_size,).
        """
        frame = visual_frame[:self.vision_size].astype(np.float32)
        self._history.append(frame)

        if len(self._history) < 3:
            self._score = 0.0
            return 0.0

        frames = list(self._history)
        curr   = frames[-1]
        prev   = frames[-2]

        # --- Feature 1: temporal autocorrelation (Troje 2002) ---
        # High r → smooth trajectory → biological (Pearson r correlation)
        delta = curr - prev
        mean_c = np.mean(curr)
        mean_p = np.mean(prev)
        var_c = float(np.var(curr) + 1e-6)
        var_p = float(np.var(prev) + 1e-6)
        cov = float(np.mean((curr - mean_c) * (prev - mean_p)))
        autocorr = cov / (np.sqrt(var_c * var_p) + 1e-6)
        # Clamp to [0, 1]: negative means jerky/reversed, not biological
        autocorr_score = float(np.clip(autocorr, 0.0, 1.0))

        # --- Feature 2: structured spatial variance (Cutting & Kozlowski 1977) ---
        # Compute per-pixel variance across history; its std indicates structure.
        # Uniform global motion → all pixels move equally → low std(variance)
        stack = np.stack(frames, axis=0)           # (T, V)
        px_var = np.var(stack, axis=0)             # (V,) per-pixel variance
        spatial_structure = float(np.std(px_var))  # high → different parts move differently
        # Normalise: typical range 0–0.15; cap at 1
        structured_score = float(np.clip(spatial_structure / 0.15, 0.0, 1.0))

        # --- Feature 3: periodicity via IIR bandpass (Grossman & Blake 2002) ---
        # Bandpass ≈ 1–4 Hz at 10 Hz tick rate: keeps periodic biological rhythms,
        # attenuates DC drift and high-freq noise.
        mean_luminance = float(np.mean(curr))
        self._bp_state = _BP_A * self._bp_state + (1.0 - _BP_A) * (curr - mean_luminance)
        periodicity_energy = float(np.mean(self._bp_state ** 2))
        periodicity_score  = float(np.clip(periodicity_energy / 0.02, 0.0, 1.0))

        # --- Combine: require all three features (AND-like product) ---
        # Pure noise: high autocorr but low structure and low periodicity → low score
        # Camera pan: high autocorr, low structure → filtered out
        # Biological: moderate autocorr, high structure, some periodicity → high score
        raw = autocorr_score * structured_score * periodicity_score
        # Smooth output to avoid single-tick spikes
        self._score = float(0.7 * self._score + 0.3 * raw)
        return self._score
