"""Biological sleep architecture: N1 → N2 → N3 → REM cycling.

Cycle durations (one full cycle ≈ 600 ticks):
  N1  30 ticks  — transition, spindle onset
  N2 170 ticks  — sleep spindles, memory indexing (no weight change)
  N3 200 ticks  — SWS: SWR replay (STDP 5×), synaptic homeostasis (SHY)
  REM 200 ticks — PGO waves, dream activity, emotional reprocessing

U-curve (Hobson 1989): from cycle 3 onward N3 shrinks 40 ticks/cycle,
REM grows 40 ticks/cycle, capped at 360 ticks. Wake only at REM exit when
sleep_pressure < 0.2. Episodic memory cleared only after all phases complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch


class SleepPhase(Enum):
    N1  = auto()
    N2  = auto()
    N3  = auto()
    REM = auto()


@dataclass
class SleepResult:
    still_sleeping:   bool
    dream_activity:   Optional[np.ndarray]  # non-None only in REM (numpy, for UI)
    dream_tensor:     object                # torch.Tensor already on device — avoids second CPU→GPU transfer
    current_phase:    SleepPhase
    neuromod_targets: dict                  # keys: acetylcholine, noradrenaline, serotonin, anandamide
    is_lucid:         bool = False          # dlPFC re-engaged during REM (LaBerge 1985)


_BASE_N3  = 200
_BASE_REM = 200
_MAX_REM  = 360
_SHY_SCALE = 0.995   # Tononi synaptic homeostasis per N3 tick
_PGO_PERIOD = 5      # ticks between PGO bursts in REM

# Lucid dreaming probability model (LaBerge 1985; Hobson 2009)
# Baseline per-tick probability in REM — calibrated so that without any
# self-monitoring habit the brain has ~1-2 lucid dreams per several hundred
# sleep sessions (matches population average of ~1-2 per year).
_LD_BASELINE_PROB = 3e-3


class SleepCycle:

    def __init__(self):
        self.is_sleeping:    bool       = False
        self.current_phase:  SleepPhase = SleepPhase.N1
        self.ticks_in_phase: int        = 0
        self.cycle_number:   int        = 0

        self._nrem_queue:      list  = []
        self._rem_pool:        list  = []
        self._swr_replay_idx:  int   = 0

        self._spindle_phase:   float = 0.0   # oscillator, ≈13 Hz proxy
        self._pgo_tick:        int   = 0
        self._rem_memory_A:    Optional[np.ndarray] = None
        self._rem_memory_B:    Optional[np.ndarray] = None
        self._pgo_noise:       Optional[np.ndarray] = None

        self._visual_mask:  Optional[np.ndarray] = None
        self._num_neurons:  int = 0
        self._vision_size:  int = 0

        self._n3_duration:  int = _BASE_N3
        self._rem_duration: int = _BASE_REM

        # Lucid dreaming state
        # ACh accumulates within a single REM session — the longer the brain
        # stays in REM, the higher the cholinergic tone, the more likely dlPFC
        # can momentarily re-engage (LaBerge 1985).
        self._rem_ach_accum:  float = 0.0
        self.is_lucid:        bool  = False   # True while current REM is lucid

        # Optional dream injector — replaces PGO noise with video frames
        self._dream_injector = None

    # ------------------------------------------------------------------
    # Public setup
    # ------------------------------------------------------------------

    def set_visual_mask(self, vision_size: int, num_neurons: int) -> None:
        self._num_neurons = num_neurons
        self._vision_size = vision_size
        mask = np.zeros(num_neurons, dtype=np.float32)
        if vision_size > 0:
            mask[:vision_size] = 1.0
        self._visual_mask = mask

    def set_dream_injector(self, injector) -> None:
        """Attach a DreamInjector — its video frames will replace PGO noise during REM."""
        self._dream_injector = injector

    def begin_sleep(self, hippocampus) -> None:
        """Called at sleep onset — populates queues, does NOT clear episodic memory."""
        self.current_phase  = SleepPhase.N1
        self.ticks_in_phase = 0
        self.cycle_number   = 0
        self._swr_replay_idx = 0
        self._spindle_phase  = 0.0
        self._pgo_tick       = 0
        self._pgo_noise      = None
        self._rem_memory_A   = None
        self._rem_memory_B   = None

        self._nrem_queue = hippocampus.retrieve_for_nrem()
        self._rem_pool   = hippocampus.retrieve_for_rem()
        self._update_cycle_durations()

        # Reset dream video to start of clip each sleep session
        if self._dream_injector is not None:
            self._dream_injector.reset()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_sleep_tick(
        self,
        hippocampus,
        plasticity,
        current_cortisol: float,
        current_activity_np: np.ndarray,
        sleep_pressure: float,
        pfc_monitoring_strength: float = 0.0,
    ) -> SleepResult:

        self.ticks_in_phase += 1

        if self.current_phase == SleepPhase.N1:
            result = self._tick_n1()
        elif self.current_phase == SleepPhase.N2:
            result = self._tick_n2()
        elif self.current_phase == SleepPhase.N3:
            result = self._tick_n3(plasticity)
        else:
            result = self._tick_rem(hippocampus, plasticity, pfc_monitoring_strength)

        still_sleeping = self._maybe_advance_phase(hippocampus, sleep_pressure)
        result.still_sleeping = still_sleeping
        return result

    # ------------------------------------------------------------------
    # Per-phase tick handlers
    # ------------------------------------------------------------------

    def _tick_n1(self) -> SleepResult:
        return SleepResult(
            still_sleeping=True,
            dream_activity=None,
            dream_tensor=None,
            current_phase=SleepPhase.N1,
            neuromod_targets={
                'acetylcholine': 0.3,
                'noradrenaline': 0.2,
                'serotonin':     0.3,
                'anandamide':    0.15,
            },
        )

    def _tick_n2(self) -> SleepResult:
        # Sleep spindle oscillator ≈ 13 Hz proxy (Sejnowski 2000)
        self._spindle_phase = (self._spindle_phase + 0.21) % (2 * np.pi)
        return SleepResult(
            still_sleeping=True,
            dream_activity=None,
            dream_tensor=None,
            current_phase=SleepPhase.N2,
            neuromod_targets={
                'acetylcholine': 0.1,
                'noradrenaline': 0.05,
                'serotonin':     0.1,
                'anandamide':    0.25,
            },
        )

    def _tick_n3(self, plasticity) -> SleepResult:
        # Sharp-wave ripple replay (Stickgold 2005)
        if self._swr_replay_idx < len(self._nrem_queue):
            episode = self._nrem_queue[self._swr_replay_idx]
            self._swr_replay_idx += 1
            device = plasticity.device
            memory_tensor = torch.tensor(
                episode['pattern'], dtype=torch.float32, device=device)
            plasticity.apply_stdp_learning(memory_tensor, neuromodulator_multiplier=5.0)

        # Synaptic homeostasis — global weight downscale (Tononi SHY 2006)
        plasticity.weights_values = plasticity.weights_values * _SHY_SCALE
        plasticity.weights_values = torch.clamp(plasticity.weights_values, -1.0, 1.0)
        plasticity._topology_changed = True

        # Call pruning periodically during N3 (every 20 ticks) to clear weak/unused synapses
        if self.ticks_in_phase % 20 == 0:
            plasticity.perform_sleep_pruning()

        return SleepResult(
            still_sleeping=True,
            dream_activity=None,
            dream_tensor=None,
            current_phase=SleepPhase.N3,
            neuromod_targets={
                'acetylcholine': 0.05,
                'noradrenaline': 0.02,
                'serotonin':     0.08,
                'anandamide':    0.2,
            },
        )

    @property
    def rem_atonia_active(self) -> bool:
        """REM atonia: motor neurons are inhibited during REM (Pelosi 2010).

        The pons (sublaterodorsal nucleus) hyperpolarises spinal motor
        neurons via glycinergic input throughout REM, preventing dream
        movements from being executed. Without this you'd act out dreams
        (REM Sleep Behaviour Disorder, Schenck 1986).

        Brain.py reads this flag and zeros motor output when True.
        """
        return self.is_sleeping and self.current_phase == SleepPhase.REM

    def _tick_rem(self, hippocampus, plasticity,
                  pfc_monitoring_strength: float = 0.0) -> SleepResult:
        n = self._num_neurons or 1
        if self._visual_mask is None:
            self._visual_mask = np.zeros(n, dtype=np.float32)

        self._pgo_tick += 1
        on_pgo_burst = (self._pgo_tick % _PGO_PERIOD == 0)

        if on_pgo_burst:
            # PGO burst: pick two episodes and generate pontine noise
            pool = self._rem_pool if self._rem_pool else self._nrem_queue
            if pool:
                ep_a = pool[np.random.randint(len(pool))]
                ep_b = pool[np.random.randint(len(pool))]
                self._rem_memory_A = ep_a['pattern'].copy() if len(ep_a['pattern']) == n else np.zeros(n, dtype=np.float32)
                self._rem_memory_B = ep_b['pattern'].copy() if len(ep_b['pattern']) == n else np.zeros(n, dtype=np.float32)

                # Emotional reprocessing: NA=0 during REM strips fear charge (Walker 2009)
                for ep in [ep_a, ep_b]:
                    if abs(ep['valence']) > 0.3:
                        hippocampus.reduce_valence(ep['id'], factor=0.85)

            if self._dream_injector is not None:
                # Injected video frame — processed through same retina pipeline
                # as waking vision. PFC is offline in REM so brain treats it as real.
                # Hobson (2009): "the brain is a belief machine; in REM it believes
                # its own internally generated signals are real."
                injected = self._dream_injector.next_frame()  # (vision_size,) float32
                vs = self._vision_size
                # Build full-neuron array: video in visual slots, sparse noise elsewhere
                base_noise = np.random.rand(n).astype(np.float32) * 0.1
                if len(injected) == vs:
                    w = self._dream_injector.blend_weight
                    base_noise[:vs] = base_noise[:vs] * (1.0 - w) + injected * w
                self._pgo_noise = base_noise
            else:
                # Natural PGO: pontine noise amplified at visual cortex (Hobson 1988)
                base_noise = np.random.rand(n).astype(np.float32) * 0.3
                base_noise += self._visual_mask * np.random.rand(n).astype(np.float32) * 0.5
                self._pgo_noise = base_noise

        if self._pgo_noise is None:
            self._pgo_noise = np.zeros(n, dtype=np.float32)

        mem_a = self._rem_memory_A if self._rem_memory_A is not None else np.zeros(n, dtype=np.float32)
        mem_b = self._rem_memory_B if self._rem_memory_B is not None else np.zeros(n, dtype=np.float32)

        # Cross-memory association: blend episodes with dream content (Stickgold 2002)
        # When injector active: video dominates visual channels, episodes add context
        dream = self._pgo_noise * self._visual_mask + mem_a * 0.6 + mem_b * 0.4
        dream_activity = np.clip(dream, 0.0, 1.0).astype(np.float32)

        # Single CPU→GPU transfer — tensor reused for both STDP and activity update
        device       = plasticity.device
        dream_tensor = torch.tensor(dream_activity, dtype=torch.float32, device=device)
        plasticity.apply_stdp_learning(dream_tensor, neuromodulator_multiplier=2.0)

        # ------------------------------------------------------------------
        # Lucid dream probability (LaBerge 1985; Hobson 2009; Voss 2009)
        #
        # dlPFC re-engages during REM when cholinergic tone is high enough
        # to partially overcome the aminergic suppression of self-monitoring.
        # Four multiplicative factors — all must be elevated simultaneously:
        #
        # 1. cycle_factor: later cycles have longer REM and more ACh build-up.
        #    Lucid dreams are ~4× more common in cycles 3-5 than cycle 1.
        #    (Green & McCreery 1994)
        #
        # 2. ach_factor: ACh accumulates within the current REM session.
        #    The longer the brain stays in REM, the more likely PFC intrusion.
        #    (Hobson/McCarley cholinergic REM-on model 1975)
        #
        # 3. monitoring_factor: strength of the waking self-monitoring habit.
        #    Any technique that trains "am I dreaming?" during waking hours
        #    strengthens the cortico-cortical dlPFC→awareness pathway.
        #    This pathway fires spontaneously in REM when strong enough.
        #    MILD, DILD, reality checks, dream journaling all work through this.
        #    (Stumbrys 2012; Erlacher 2008)
        #
        # 4. anomaly_factor: dream content anomaly score — impossible or
        #    contradictory elements create prediction error that can pierce
        #    through the PFC suppression. High cross-memory blending (large
        #    difference between mem_a and mem_b) = more impossible dream content.
        #    (Hobson "AIM model" 2009 — activation-input-modulation)
        # ------------------------------------------------------------------

        # ACh accumulates each REM tick, resets at REM exit
        self._rem_ach_accum = min(1.0, self._rem_ach_accum + 0.008)

        # Factor 1: cycle number (cycles 0-1 = baseline, cycles 3+ = 4×)
        cycle_factor = min(4.0, 1.0 + max(0, self.cycle_number - 1) * 1.0)

        # Factor 2: ACh within this REM session
        ach_factor = self._rem_ach_accum ** 2   # quadratic: rare until saturated

        # Factor 3: waking self-monitoring habit (passed from brain.py)
        # Decays in [0,1] — zero = no habit, 1 = strong daily practice
        monitoring_factor = 1.0 + pfc_monitoring_strength * 40.0

        # Factor 4: dream content anomaly — how impossible is this dream?
        # High variance in the blended memories = contradictory content
        anomaly = float(np.std(dream_activity))
        anomaly_factor = min(3.0, anomaly / 0.1)

        lucid_prob = (_LD_BASELINE_PROB
                      * cycle_factor
                      * ach_factor
                      * monitoring_factor
                      * anomaly_factor)

        triggered_lucid = (not self.is_lucid) and (np.random.random() < lucid_prob)
        if triggered_lucid:
            self.is_lucid = True

        # Lucid state neuromodulation: partial dlPFC re-engagement
        # NA rises slightly (0.0 → ~0.15) — not enough to wake, enough for
        # metacognition. ACh stays high. This is the "hybrid state" (Voss 2009).
        if self.is_lucid:
            na_target    = 0.15
            ach_target   = 0.95   # even higher than normal REM
            serotonin_target = 0.05  # tiny raphe activity — supports self-awareness
        else:
            na_target    = 0.0
            ach_target   = 0.9
            serotonin_target = 0.0

        return SleepResult(
            still_sleeping=True,
            dream_activity=dream_activity,
            dream_tensor=dream_tensor,
            current_phase=SleepPhase.REM,
            is_lucid=self.is_lucid,
            neuromod_targets={
                'acetylcholine': ach_target,
                'noradrenaline': na_target,
                'serotonin':     serotonin_target,
                'anandamide':    0.7,
            },
        )

    # ------------------------------------------------------------------
    # Phase advancement
    # ------------------------------------------------------------------

    def _maybe_advance_phase(self, hippocampus, sleep_pressure: float) -> bool:
        durations = {
            SleepPhase.N1:  30,
            SleepPhase.N2:  170,
            SleepPhase.N3:  self._n3_duration,
            SleepPhase.REM: self._rem_duration,
        }
        if self.ticks_in_phase < durations[self.current_phase]:
            return True  # stay in phase

        self.ticks_in_phase = 0
        if self.current_phase == SleepPhase.N1:
            self.current_phase = SleepPhase.N2
        elif self.current_phase == SleepPhase.N2:
            self.current_phase = SleepPhase.N3
            self._swr_replay_idx = 0   # reset replay cursor for this N3 bout
        elif self.current_phase == SleepPhase.N3:
            self.current_phase = SleepPhase.REM
            self._pgo_tick = 0
        elif self.current_phase == SleepPhase.REM:
            self.cycle_number += 1
            self._update_cycle_durations()
            # Reset REM-local state
            self._rem_ach_accum = 0.0
            self.is_lucid       = False
            # Natural wake only at REM exit when sleep pressure dissipated
            if sleep_pressure < 0.2:
                hippocampus.episodic_memory.clear()
                self.is_sleeping = False
                return False
            # Continue into next cycle
            self.current_phase   = SleepPhase.N1
            self._swr_replay_idx = 0
            self._pgo_tick       = 0

        return True

    def _update_cycle_durations(self) -> None:
        """U-curve: SWS dominant early, REM dominant late (Hobson 1989)."""
        shrink = max(0, self.cycle_number - 2) * 40
        self._n3_duration  = max(40,       _BASE_N3  - shrink)
        self._rem_duration = min(_MAX_REM, _BASE_REM + shrink)
