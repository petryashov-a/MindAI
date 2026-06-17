"""Anterior Cingulate Cortex (ACC) — conflict monitoring and error detection.

Biological basis (Botvinick et al. 2001; Carter & van Veen 2007):
  The dACC (dorsal ACC, area 24/32) monitors response conflict — the simultaneous
  activation of competing motor programs — and signals the need for increased
  cognitive control.

  Error-related negativity (ERN, Falkenstein 1991; Gehring 1993):
    ~80–100 ms after an error, a large negative deflection in the EEG is
    recorded over frontal midline.  Source localisation: dACC.  This is the
    first brain marker that "something went wrong" — before the person is
    consciously aware of the error.

  Conflict monitoring hypothesis (Botvinick 2001):
    ACC monitors for simultaneous activation of mutually exclusive responses
    (e.g. Simon task: spatial location says LEFT, word says RIGHT).
    High conflict → signals dlPFC to increase top-down control.

  Role in pain affect (Rainville 1997):
    ACC encodes the *unpleasantness* of pain (affective dimension), not
    the sensation intensity (encoded in S1/S2).  Hypnotic suggestion can
    reduce ACC activation and unpleasantness without changing S1 activation.

  Output:
    → dlPFC: request increased executive control
    → Amygdala: conflict-induced anxiety
    → Striatum: conflict-induced motivational adjustment

Implementation:
  Conflict index = sum of products of competing action probabilities.
  For action vector p: conflict = Σᵢ Σⱼ≠ᵢ pᵢ × pⱼ = 1 − Σᵢ pᵢ²
  (Botvinick 2001 eq. 2 — normalised Hopfield energy).

  Error detection: compares chosen action motor output with the prediction
  from the previous tick.  Sustained error → ERN signal.
"""

from __future__ import annotations
import numpy as np


class AnteriorCingulate:

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

        # Running conflict level (smoothed)
        self.conflict: float = 0.0

        # ERN — error-related negativity signal
        self.ern: float = 0.0

        # Affective pain unpleasantness (separate from S1 intensity)
        self.pain_unpleasantness: float = 0.0

        # Executive control request to dlPFC [0,1]
        self.control_request: float = 0.0

        # Previous action distribution for comparison
        self._prev_action_probs: np.ndarray = np.ones(num_actions) / num_actions

    def update(
        self,
        action_probs:   np.ndarray,  # softmax motor probabilities
        chosen_action:  int | None,
        prediction_error: float,     # from PredictiveMicrocircuits
        pain_signal:    float,       # nociceptive input (amygdala threat or raw)
    ) -> dict:
        """Compute conflict, ERN, and control request for one tick.

        Returns:
          conflict         : float [0,1] — response conflict
          ern              : float [0,1] — error-related negativity
          control_request  : float [0,1] — signal to dlPFC/PFC
          pain_unpleasantness : float [0,1] — affective pain (not intensity)
        """
        p = action_probs[:self.num_actions]
        if p.sum() > 1e-9:
            p = p / p.sum()

        # Conflict index (Botvinick 2001): 1 − Σpᵢ²
        # Max = 1 − 1/N (uniform); min = 0 (single action dominates)
        raw_conflict = 1.0 - float(np.sum(p ** 2))
        self.conflict = float(0.8 * self.conflict + 0.2 * raw_conflict)

        # ERN: mismatch between chosen action and previous action distribution
        # Models the rapid post-error negativity (~100 ms latency)
        if chosen_action is not None and chosen_action < self.num_actions:
            expected_prob = float(self._prev_action_probs[chosen_action])
            error_signal  = max(0.0, expected_prob - p[chosen_action])
            self.ern = float(np.clip(0.7 * self.ern + 0.3 * error_signal, 0.0, 1.0))
        else:
            self.ern *= 0.8

        # Affective pain unpleasantness — ACC encodes "how bad it feels"
        # separated from sensory intensity (Rainville 1997)
        # Rises with pain AND with conflict (pain × conflict = worst)
        self.pain_unpleasantness = float(np.clip(
            0.9 * self.pain_unpleasantness
            + 0.1 * pain_signal * (1.0 + self.conflict),
            0.0, 1.0))

        # Control request to dlPFC: driven by conflict + ERN + prediction error
        raw_control = (self.conflict * 0.5
                       + self.ern * 0.3
                       + min(1.0, prediction_error * 0.05) * 0.2)
        self.control_request = float(np.clip(
            0.85 * self.control_request + 0.15 * raw_control, 0.0, 1.0))

        self._prev_action_probs[:] = p

        return {
            'conflict':              self.conflict,
            'ern':                   self.ern,
            'control_request':       self.control_request,
            'pain_unpleasantness':   self.pain_unpleasantness,
        }
