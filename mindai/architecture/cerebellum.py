"""Cerebellum — predictive motor control via climbing-fibre error signals.

Biological basis (Ito 1984; Wolpert & Kawato 1998):
  The cerebellum operates as a forward model: it predicts the sensory
  consequences of a motor command BEFORE execution.  When reality differs
  from prediction, climbing fibres from the inferior olive carry a teaching
  signal (complex spike) that drives long-term depression (LTD) at
  parallel fibre → Purkinje cell synapses.

  Key components:
    Granule cells  — expand motor/proprioceptive context into high-dimensional
                     sparse representation (Marr 1969; Albus 1971).
    Purkinje cells — learn to predict the expected reafference vector.
                     Output suppresses cerebellar nuclei (inhibitory).
    Climbing fibres (inferior olive) — one per Purkinje cell; fire only when
                     prediction error exceeds a threshold.  Rate ≤ 1–2 Hz
                     (sluggish compared to 200 Hz mossy-fibre background).
    Deep cerebellar nuclei (DCN) — tonic excitatory output to thalamus/spinal
                     cord; gated by Purkinje inhibition.

  Learning rule (Marr-Albus-Ito):
    When climbing fibre fires (error > θ):
      Δw = −η × granule_active × (prediction_error)    [LTD]
    When no error (climbing fibre silent):
      Δw = +η_slow × granule_active                    [slow LTP, rebound]

  Timing prediction — internal model of limb dynamics.
    The cerebellum encodes expected delay between motor command and
    proprioceptive feedback (~50–150 ms for arm, shorter for eye).
    Here approximated as: predicted_reafference[t] = W × motor_command[t-1].
    Error = actual_reafference[t] − predicted.

No gradient descent, no reward.  Climbing-fibre LTD is purely Hebbian:
the motor error IS the teaching signal, not a scalar reward shaped by a
developer.
"""

from __future__ import annotations
import numpy as np


_GRANULE_EXPANSION = 4     # expansion ratio: motor_size × 4 = granule count
_LTD_RATE          = 0.02  # parallel fibre LTD per climbing-fibre spike
_LTP_RATE          = 0.002 # slow rebound LTP when no error
_CF_THRESHOLD      = 0.08  # climbing-fibre firing threshold (prediction error magnitude)


class Cerebellum:

    def __init__(self, motor_size: int, reafference_size: int):
        """
        motor_size:       number of motor output neurons (from motor cortex)
        reafference_size: number of proprioceptive/reafference input channels
        """
        self.motor_size       = motor_size
        self.reafference_size = reafference_size
        self.granule_size     = motor_size * _GRANULE_EXPANSION

        # Granule cell encoding matrix — random sparse projections (Marr 1969)
        # Biologically scaled connectivity to prevent dead cells in small networks
        rng = np.random.default_rng(42)
        sparsity = max(0.05, 3.0 / motor_size)
        self._granule_w = (rng.random((self.granule_size, motor_size)) < sparsity).astype(np.float32)

        # Purkinje cell weights: granule → reafference prediction
        self._purkinje_w = np.zeros(
            (reafference_size, self.granule_size), dtype=np.float32)

        # Previous motor command (1-tick delay = forward model)
        self._prev_motor: np.ndarray = np.zeros(motor_size, dtype=np.float32)

        # Predicted reafference from last tick
        self.predicted_reafference: np.ndarray = np.zeros(reafference_size, dtype=np.float32)

        # Smoothed prediction error — exposed for brain.py
        self.prediction_error: float = 0.0

        # DCN output: smoothed excitatory drive to thalamus
        self.dcn_output: float = 0.0

        # Climbing fiber refractory period counter (Ito 1984)
        self._cf_cooldown: int = 0

    def update(
        self,
        motor_command:    np.ndarray,   # current motor activity vector
        actual_reafference: np.ndarray, # proprioceptive / sensory feedback
    ) -> dict:
        """One tick of cerebellar computation.

        Returns:
          prediction_error : float — magnitude of prediction mismatch
          dcn_output       : float — excitatory drive to motor thalamus
          correction       : np.ndarray — additive correction to motor command
        """
        mc  = motor_command[:self.motor_size]
        rea = actual_reafference[:self.reafference_size]

        # Granule cell activation (sparse, high-dimensional context)
        granule = (self._granule_w @ self._prev_motor) > 0.0

        # Purkinje prediction
        pred = self._purkinje_w @ granule.astype(np.float32)
        self.predicted_reafference = pred

        # Prediction error (what the climbing fibres encode)
        error_vec = rea - pred
        error_mag = float(np.mean(np.abs(error_vec)))
        self.prediction_error = float(0.8 * self.prediction_error + 0.2 * error_mag)

        # Climbing-fibre LTD / LTP
        climbing_fire = (error_mag > _CF_THRESHOLD) and (self._cf_cooldown <= 0)
        if self._cf_cooldown > 0:
            self._cf_cooldown -= 1

        g = granule.astype(np.float32)
        if climbing_fire:
            self._cf_cooldown = 5  # Refractory period: ~5 ticks (500ms at 10Hz)
            # LTD: reduce Purkinje weights for active granule cells
            delta = -_LTD_RATE * np.outer(error_vec, g)
            self._purkinje_w = np.clip(self._purkinje_w + delta, -1.0, 1.0)
        else:
            # Slow LTP rebound (Boyden 2004)
            self._purkinje_w = np.clip(
                self._purkinje_w + _LTP_RATE * np.outer(np.ones(self.reafference_size), g),
                -1.0, 1.0)

        # DCN: Purkinje cells are inhibitory → DCN fires when Purkinje is silent
        purkinje_norm = float(np.mean(np.abs(pred)))
        self.dcn_output = float(np.clip(1.0 - purkinje_norm, 0.0, 1.0))

        # Motor correction: DCN → motor thalamus → M1 additive signal
        # When error is large, DCN corrects the ongoing command
        correction = np.zeros(self.motor_size, dtype=np.float32)
        if climbing_fire:
            # Project error vector back through Purkinje weights and granule weights to motor space
            error_proj = (self._granule_w.T @ ((self._purkinje_w.T @ error_vec) * g)).astype(np.float32)
            correction = np.clip(error_proj * 0.1, -0.3, 0.3)

        self._prev_motor[:] = mc

        return {
            'prediction_error': self.prediction_error,
            'dcn_output':       self.dcn_output,
            'correction':       correction,
        }
