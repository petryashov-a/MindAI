"""Amygdala — fear conditioning and threat detection.

Biological basis:
  The amygdala has two parallel input pathways (LeDoux 1996):

  Fast path (thalamo-amygdalar, ~12 ms):
    Thalamus → lateral nucleus directly.
    Coarse, low-resolution signal. Triggers immediate defensive response
    before cortical processing is complete. Biologically critical: allows
    freezing/fleeing before consciously recognising the threat.

  Slow path (thalamo-cortico-amygdalar, ~40–50 ms):
    Thalamus → sensory cortex → lateral nucleus.
    High-resolution, context-rich signal. Can override or confirm fast path.

  Fear conditioning (Pavlov/LeDoux):
    CS (neutral stimulus) co-activates with US (pain/threat) in lateral nucleus.
    Hebbian LTP strengthens CS→lateral synapse.
    After conditioning: CS alone triggers full fear response.

  Extinction (Milad & Quirk 2002):
    Repeated CS without US → vmPFC inhibits central nucleus → fear decreases.
    Modelled as slow decay of CS→lateral weights when US absent.

  Output — central nucleus (CeA):
    → Hypothalamus: autonomic arousal (heart rate, adrenaline)
    → LC: noradrenaline surge
    → VTA: dopamine modulation (both excitation for salience and indirect suppression via RMTg)
    → Striatum: avoidance bias

  Basolateral nucleus (BLA):
    Emotional memory tagging — co-activates with hippocampus to
    strengthen episodic memories with high emotional valence.
"""

from __future__ import annotations
import numpy as np


_FAST_PATH_LATENCY  = 1   # ticks (thalamic direct, ~12 ms at 10 Hz)
_SLOW_PATH_LATENCY  = 4   # ticks (cortico-amygdalar, ~40 ms at 10 Hz)
_LTP_RATE           = 0.05
_EXTINCTION_RATE    = 0.002   # slow — consistent with clinical extinction curves
_THREAT_DECAY       = 0.85    # per tick when no US present


class Amygdala:

    def __init__(self, num_sensory: int):
        """
        num_sensory: size of the sensory input vector (vision + audio + pain).
        """
        self.num_sensory = num_sensory

        # Lateral nucleus: CS → fear association weights
        # Initialised near zero — all fear must be learned (LeDoux 1996)
        self.cs_weights_fast = np.zeros(num_sensory, dtype=np.float32)
        self.cs_weights_slow = np.zeros(num_sensory, dtype=np.float32)

        # Current threat level output from central nucleus (CeA)
        self.threat_level: float = 0.0

        # BLA emotional tagging signal (read by hippocampus for memory salience)
        self.emotional_tag: float = 0.0

        # Extinction counter: ticks since last US co-activation
        self._no_us_ticks: int = 0

        # Fast path buffer: thalamic signal arrives 1 tick after sensory input
        self._fast_buf: np.ndarray = np.zeros(num_sensory, dtype=np.float32)

    def update(
        self,
        sensory_input: np.ndarray,    # raw sensory vector (vision+audio+pain)
        pain_signal:   float,         # unconditioned stimulus (US)
        dopamine:      float,         # VTA dopamine (low = aversive context)
    ) -> dict:
        """Process one tick. Returns dict of output signals for brain.py.

        Returns:
          threat_level    : float [0,1] — CeA output, drives NA/adrenaline
          emotional_tag   : float [0,1] — BLA tagging for hippocampal encoding
          da_suppression  : float [0,1] — aversive prediction error → DA drop
          fear_conditioned: bool  — CS alone triggered fear this tick
        """
        s = sensory_input[:self.num_sensory]

        # --- Fast path (1-tick latency thalamo-amygdalar) ---
        fast_activation = float(np.dot(self._fast_buf, self.cs_weights_fast))
        self._fast_buf[:] = s   # store for next tick

        # --- Slow path (current tick, cortical CS representation) ---
        slow_activation = float(np.dot(s, self.cs_weights_slow))

        # --- Unconditioned stimulus (US = pain) ---
        us_present = pain_signal > 0.3

        if us_present:
            self._no_us_ticks = 0
            # Hebbian LTP: CS neurons co-active with US → strengthen weights
            # Only update for active sensory channels
            active = (s > 0.2).astype(np.float32)
            delta_fast = _LTP_RATE * active * pain_signal * (1.0 - self.cs_weights_fast)
            self.cs_weights_fast = np.clip(self.cs_weights_fast + delta_fast, 0.0, 1.0)
            
            delta_slow = _LTP_RATE * active * pain_signal * (1.0 - self.cs_weights_slow)
            self.cs_weights_slow = np.clip(self.cs_weights_slow + delta_slow, 0.0, 1.0)
        else:
            self._no_us_ticks += 1
            # Extinction: vmPFC inhibits CeA when CS presented without US
            # Slow decay — extinction is fragile (Milad & Quirk 2002)
            if self._no_us_ticks > 50:
                active = (s > 0.2).astype(np.float32)
                self.cs_weights_fast = np.clip(
                    self.cs_weights_fast - _EXTINCTION_RATE * active, 0.0, 1.0)
                self.cs_weights_slow = np.clip(
                    self.cs_weights_slow - _EXTINCTION_RATE * active, 0.0, 1.0)

        # --- Central nucleus activation ---
        # Fast path provides the immediate threat signal; slow path refines
        raw_threat = max(fast_activation, slow_activation * 0.8)
        if us_present:
            raw_threat = max(raw_threat, pain_signal)

        self.threat_level = float(np.clip(
            self.threat_level * _THREAT_DECAY + raw_threat * 0.3, 0.0, 1.0))

        fear_conditioned = (not us_present) and (raw_threat > 0.3)

        # --- BLA emotional tagging ---
        # Hippocampal memories formed during high amygdala activation or reward (DA)
        # are consolidated preferentially (McGaugh 2004; Hamann 2001)
        self.emotional_tag = float(np.clip(self.threat_level * 1.5 + dopamine * 0.5, 0.0, 1.0))

        # --- DA suppression (aversive PE) ---
        # High threat + low dopamine = strong aversive signal
        da_suppression = float(np.clip(self.threat_level * (1.0 - dopamine), 0.0, 1.0))

        return {
            'threat_level':     self.threat_level,
            'emotional_tag':    self.emotional_tag,
            'da_suppression':   da_suppression,
            'fear_conditioned': fear_conditioned,
        }
