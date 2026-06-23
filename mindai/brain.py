"""Brain — biologically-inspired artificial mind.

Core design
-----------
The Brain is the substrate.  Everything else is optional:

    from mindai import Brain
    from mindai.worlds.agent_world import AgentWorld
    from mindai.neurochemistry.neuromodulators import EndocrineSystem

    world = AgentWorld(text_corpus='corpus.txt')

    # Minimal brain (works alone — no feelings, no hormones)
    brain = Brain(num_neurons=500_000,
                  sensory_layout=world.sensory_layout,
                  motor_layout=world.motor_layout)
    brain.run(world, headless=True)

    # With optional feelings (psychophysical curves over world signals)
    from mindai.feels import FeelingSystem, Feel, curves
    feels = FeelingSystem()
    feels.add(Feel('pain',   'pain',   curve=curves.power(1.5)))
    feels.add(Feel('hunger', 'hunger', curve=curves.quadratic))
    brain.attach(feels)

    # With optional hormones
    brain.attach(EndocrineSystem())

    brain.run(world)

All learning is Hebbian/STDP — no gradient descent, no reward function.
Behaviour emerges from physiology gated by feelings and neuromodulators.

Save format
-----------
Brain state is saved to a *directory* (separate from world state):

    save_dir/
        brain.json      tick, num_neurons, metadata
        weights.npz     sparse synapse matrix (row, col, w, integrity)

World state is saved separately by the user's world connector code.
This allows the brain to be migrated to a different world while retaining
learned synaptic structure — exactly as a biological brain retains memories
when moved to a new environment.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from mindai.layout import SensoryLayout
from mindai.engine.plasticity_core import StructuralPlasticity
from mindai.engine.spatial_topology_3d import BrainGeometry
from mindai.engine.temporal_windows import HusserlianTime
from mindai.architecture.predictive_hierarchy import PredictiveMicrocircuits
from mindai.architecture.thalamocortical_core import Thalamus
from mindai.architecture.hippocampus_buffer import Hippocampus
from mindai.architecture.prefrontal_cortex import PrefrontalCortex
from mindai.architecture.semantic_memory import SemanticMemory
from mindai.architecture.cortical_layers import CorticalLayers
from mindai.architecture.cortical_areas import CorticalAreas
from mindai.architecture.biological_motion_detector import BiologicalMotionDetector
from mindai.architecture.cerebellum import Cerebellum
from mindai.architecture.anterior_cingulate import AnteriorCingulate
from mindai.architecture.insula import Insula
from mindai.architecture.default_mode_network import DefaultModeNetwork
from mindai.architecture.visuospatial_sketchpad import VisuospatialSketchpad
from mindai.architecture.theory_of_mind import TheoryOfMind
from mindai.engine.axonal_delays import DelayQueue, build_delay_tensor
from mindai.consciousness.global_workspace import PhaseCoupledWorkspace
from mindai.consciousness.self_model_ego import EgoModel
from mindai.consciousness.volition_and_agency import FreeWillEngine, BasalGanglia
from mindai.consciousness.neural_complexity import NeuralComplexity
from mindai.architecture.amygdala import Amygdala
from mindai.architecture.habenula import Habenula
from mindai.architecture.pag import PAG
from mindai.architecture.astrocytes import Astrocytes
from mindai.architecture.hippocampus_subfields import HippocampalSubfields
from mindai.architecture.entorhinal import EntorhinalGrid
from mindai.architecture.olfactory_bulb import OlfactoryBulb
from mindai.lifecycle.sleep_consolidation import SleepCycle, SleepPhase
from mindai.lifecycle.circadian_rhythm import BiologicalClock
from mindai.architecture.superior_colliculus import SuperiorColliculus
from mindai.architecture.language_cortex import LanguageCortex, _COMM_DRIVE_THRESHOLD
from mindai.architecture.prefrontal_cortex import DialogueWorkingMemory
from mindai.engine.sdr_encoder import SDREncoder
from mindai.engine.cross_modal_binder import CrossModalBinder


# ---------------------------------------------------------------------------
# Null-object fallbacks for optional modules
# ---------------------------------------------------------------------------

class _NullChemistry:
    """Neutral chemistry — returned when no EndocrineSystem is attached.

    All values represent a resting, non-stressed organism.  The brain functions
    without hormones but learns more slowly and without emotional gating.
    This is biologically equivalent to a decerebrate preparation: neural
    activity continues, plasticity is uniform, motivation is absent.
    """
    dopamine               = 0.5
    noradrenaline          = 0.5
    serotonin              = 0.5
    dopamine_mesocortical  = 0.5
    cortisol               = 0.0
    oxytocin               = 0.0
    adrenaline             = 0.0
    endorphins             = 0.0
    boredom                = 0.0
    acetylcholine          = 0.5
    anandamide             = 0.0
    substance_p            = 0.0
    ghrelin                = 0.0
    leptin                 = 0.5
    vasopressin            = 0.0
    prolactin              = 0.0
    insulin                = 0.0
    effective_pain_signal  = 0.0
    effective_hunger_signal= 0.0
    hippocampal_salience_gate = 1.0
    mirror_neuron_amplifier   = 0.5

    def get_plasticity_multiplier(self): return 1.0
    def update_state(self, **kw): pass
    def trigger_social_bonding(self): pass
    def trigger_endorphin_rush(self): pass
    def derive_mood(self): return 'neutral'


_NULL_CHEM = _NullChemistry()


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------

class Brain:
    """Biologically-inspired artificial mind.

    Parameters
    ----------
    num_neurons:
        Total neuron count.
    sensory_layout:
        Dict of channel_name → int (size).  Special key ``'vision'`` is always
        placed first at index 0.  Remaining channels are placed contiguously.
    motor_layout:
        Dict of channel_name → int or (start, end) tuple.  ``'motor'`` and
        ``'vocalization'`` are the standard keys; ``'mirror_neurons'`` can be
        a relative-offset tuple inside the motor block.
    device:
        ``'auto'``, ``'cpu'``, or ``'cuda'``.
    save_path:
        Path to save the brain as a directory (e.g. ``'savegame/'``).
    num_actions:
        Number of distinct motor actions (fed to BasalGanglia).
    """

    def __init__(
        self,
        num_neurons:         int,
        sensory_layout:      dict,
        motor_layout:        dict,
        device:              str   = 'auto',
        save_path:           str   = 'savegame',
        num_actions:         int   = 5,
        synapse_density:     float = 0.01,
        rehab_ticks:         int   = 0,
        k_candidates:        int   = 8,
        binder_text_dim:     int | None = None,
        episode_top_k:       int   = 1024,
    ) -> None:
        if device == 'auto':
            if torch.cuda.is_available():
                # Estimate synapse memory: (N*0.2)^2 * density * 32 bytes
                needed_gb = (num_neurons * 0.2) ** 2 * synapse_density * 32 / 1e9
                free_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.85
                if needed_gb <= free_gb:
                    self._device = torch.device('cuda')
                else:
                    print(f'>>> VRAM {free_gb:.1f} GB < needed {needed_gb:.1f} GB -> CPU')
                    self._device = torch.device('cpu')
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device)

        self.save_path = save_path
        self._rehab_ticks_left = rehab_ticks
        if rehab_ticks > 0:
            print(f'>>> [REHAB] Rehabilitating mode activated for {rehab_ticks} ticks.')

        self._layout = SensoryLayout.from_channels(
            vision_size=int(sensory_layout.get('vision', 0)),
            sensory={k: v for k, v in sensory_layout.items() if k != 'vision'},
            motor=motor_layout,
            num_neurons=num_neurons,
        )

        torch.set_grad_enabled(False)
        self.num_neurons = num_neurons
        print(f'>>> Mind initialized with {num_neurons} neurons on {self._device}')

        # Core neural substrate — always present
        self._geometry      = BrainGeometry(num_neurons)
        self._geometry.optimize_spatial_locality(self._layout)
        self._plasticity    = StructuralPlasticity(
            num_neurons,
            initial_density=synapse_density,
            device=self._device,
            coordinates=self._geometry.coordinates,
            k_candidates=k_candidates,
        )
        # Free init temporaries before next heavy allocation
        gc.collect()
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
        self._time          = HusserlianTime(num_neurons, window_size=3, device=self._device)
        pred_density = min(0.005, 12.0 / num_neurons)
        self._predictor     = PredictiveMicrocircuits(num_neurons, initial_density=pred_density, device=self._device, max_fan_in=16)
        # Free _build temporaries
        gc.collect()
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
        self._workspace     = PhaseCoupledWorkspace(num_neurons, self._device)
        self._thalamus      = Thalamus(num_neurons, self._device)
        self._ego           = EgoModel(expected_baseline=1.0)
        self._clock         = BiologicalClock(cycle_length_ticks=1500)
        self._sleep         = SleepCycle()
        vision_size = int(sensory_layout.get('vision', 0))
        self._sleep.set_visual_mask(vision_size, num_neurons)
        # Superior Colliculus — only meaningful when vision is present
        self._sc: SuperiorColliculus | None = (
            SuperiorColliculus() if vision_size > 0 else None
        )
        self._hippocampus   = Hippocampus(episode_top_k=episode_top_k)
        # New biology — anti-reward, defence switch, glia, hippocampal subfields,
        # grid/place coding, olfaction
        self._habenula      = Habenula()
        self._pag           = PAG()
        self._astrocytes    = Astrocytes(device=self._device)
        self._hipp_subf     = HippocampalSubfields()
        self._entorhinal    = EntorhinalGrid()
        self._olfactory     = OlfactoryBulb()
        self._free_will     = FreeWillEngine(delay_ticks=3)
        self._pfc           = PrefrontalCortex(num_neurons, self._layout.end('motor'))
        self._semantics     = SemanticMemory()
        self._nc            = NeuralComplexity(num_neurons)
        # Amygdala — fear conditioning (LeDoux 1996); receives sensory slice
        _sensory_size = sum(sensory_layout.values())
        self._amygdala = Amygdala(num_sensory=min(_sensory_size, num_neurons))
        self._basal_ganglia = BasalGanglia(
            motor_cortex_size=self._layout.size('motor'),
            num_actions=num_actions,
        )

        # Biological motion detector — MT+/V5 analog (Grossman & Blake 2002)
        self._motion_detector = BiologicalMotionDetector(
            vision_size=int(sensory_layout.get('vision', 0)))

        # Cortical areas — functional specialisation map (Zeki 1978)
        self._areas = CorticalAreas(num_neurons)

        # Cerebellum — forward model, climbing-fibre error (Ito 1984)
        _motor_size = self._layout.size('motor') if self._layout.has('motor') else 10
        _rea_size   = int(sensory_layout.get('vision', 100))
        self._cerebellum = Cerebellum(motor_size=_motor_size, reafference_size=_rea_size)

        # ACC — conflict monitoring + ERN (Botvinick 2001)
        self._acc = AnteriorCingulate(num_actions=num_actions)

        # Insula — interoception + body-state valence (Craig 2002)
        self._insula = Insula()

        # Default Mode Network — self-referential thought (Raichle 2001)
        self._dmn = DefaultModeNetwork(pattern_size=num_neurons)

        # Visuospatial sketchpad — spatial working memory (Baddeley 1986)
        _vw = getattr(self, '_vision_w', 24)
        self._visuospatial = VisuospatialSketchpad(vision_width=_vw, vision_height=_vw)

        # Theory of Mind — other-agent mental state inference (Saxe 2003)
        self._tom = TheoryOfMind()

        # LanguageCortex — Wernicke (comprehension) + Broca (production + syntax)
        # token_dim: SDR output size = 2% of num_neurons, min 64
        _lang_token_dim = max(64, int(num_neurons * 0.02))
        _lang_grid_dim  = self._entorhinal.total_cells
        self._language_cortex = LanguageCortex(
            token_dim    = _lang_token_dim,
            grid_dim     = _lang_grid_dim,
            semantic_dim = 128,
            vocab_size   = 256,
        )

        # SDR Encoder — sparse distributed representation for text column
        self._sdr_encoder = SDREncoder(n=_lang_token_dim)

        # Cross-modal binder — Damasio convergence zones (Damasio 1989)
        _audio_dim  = int(sensory_layout.get('audio',  64))
        _vision_dim = int(sensory_layout.get('vision', 64))
        self._lang_audio_dim  = _audio_dim
        self._lang_vision_dim = _vision_dim
        self._binder_text_dim = binder_text_dim if binder_text_dim is not None else _lang_token_dim
        self._cross_modal_binder = CrossModalBinder(modality_dims={
            'text':        self._binder_text_dim,
            'audio':       _audio_dim,
            'vision':      _vision_dim,
            'interoception': 4,   # [pain, hunger, thirst, arousal]
        })

        # Attach DialogueBeliefState to ToM (linguistic Theory of Mind)
        self._tom.attach_dialogue_belief(semantic_dim=128)

        # dlPFC Dialogue Working Memory — attached to PrefrontalCortex (BA46/9)
        # Goldman-Rakic (1995): persistent firing maintains dialogue context online.
        self._pfc.attach_dialogue_wm(semantic_dim=128)

        # Cortical layers — canonical microcircuit (Douglas & Martin 1991)
        self._layers = CorticalLayers(num_neurons, self._device)
        self._layers.apply_canonical_bias(self._plasticity)

        # Axonal delays — GPU ring buffer (Swadlow 1985)
        self._delay_queue = DelayQueue(
            num_neurons, max_delay_ticks=20, device=self._device)
        self._base_delays = build_delay_tensor(
            self._plasticity.indices[0],
            self._plasticity.indices[1],
            self._geometry.coordinates,
            device=self._device,
        )
        self.update_myelinated_delays()

        self._activity = torch.zeros(num_neurons, device=self._device)

        # Phonological loop — vocal echo feedback buffer (Baddeley 1986)
        # The brain hears its own vocalizations with a ~150–300 ms delay
        # (articulatory loop: Broca → supramarginal gyrus → auditory cortex).
        # At 10 Hz this is 2–3 ticks. Implemented as a ring buffer of depth 3.
        # Decay 0.7× per tick: echo is quieter than the original (air dampening).
        # This gives STDP a pre→post temporal window: the vocal motor output
        # (pre) precedes the auditory echo (post) by 2 ticks — STDP can now
        # strengthen the vocal→auditory path, enabling self-monitoring.
        _voc_echo_size = int(motor_layout.get('vocalization', 0))
        if isinstance(_voc_echo_size, tuple):
            _voc_echo_size = _voc_echo_size[1] - _voc_echo_size[0]
        if isinstance(_voc_echo_size, float):
            _voc_echo_size = max(4, int(round(_voc_echo_size * num_neurons)))
        self._voc_echo_buf: list[np.ndarray] = [
            np.zeros(max(1, _voc_echo_size), dtype=np.float32)
            for _ in range(3)   # 3-tick ring buffer
        ]
        self._voc_echo_idx: int = 0

        # PFC self-monitoring strength — substrate of lucid dream techniques.
        # Accumulates when the waking brain frequently encounters high surprise
        # while PFC goal is active (= the brain habitually questions its state).
        # This is the biological pathway that MILD, DILD, and reality-check
        # techniques strengthen — without scripting any specific technique.
        # Decay τ ≈ 5000 ticks (roughly days-scale at 10 Hz).
        self._pfc_monitoring_strength: float = 0.0

        # Pinned CPU buffer for zero-copy GPU transfer each tick.
        # pin_memory() allocates page-locked RAM — the DMA engine can copy
        # directly to GPU without an intermediate staging copy (2–4× faster
        # on PCIe than pageable memory, per PyTorch docs).
        if self._device.type == 'cuda':
            self._raw_pinned = torch.zeros(num_neurons, dtype=torch.float32).pin_memory()
        else:
            self._raw_pinned = None   # CPU path: no pin needed

        # Optional attached modules (None until attach() is called)
        self._chemistry: object = None   # EndocrineSystem or None
        self._feelings:  object = None   # FeelingSystem or None

        # Public simulation state
        self.tick:     int   = 0
        self.mood:     str   = 'calm'
        self.surprise: float = 0.0
        self.wellbeing: float = 1.0

        # Metabolic rate scale for CircadianRhythm.
        # Default 1.0 = normal body. Set lower (e.g. 0.1) for text-only mode
        # where motor activity is high but no real metabolic cost exists.
        self._clock_energy_scale: float = 1.0

    # -------------------------------------------------------------------------
    # Module attachment
    # -------------------------------------------------------------------------

    def attach(self, module) -> 'Brain':
        """Attach an optional module to the brain.

        Accepts any of:

        * ``FeelingSystem`` — psychophysical curves over world signals
        * ``EndocrineSystem`` — neuromodulators (dopamine, cortisol, etc.)
        * ``MoodAttractors`` — emotional attractor dynamics

        Returns self for chaining::

            brain.attach(feels).attach(EndocrineSystem())
        """
        from mindai.feels.system import FeelingSystem
        from mindai.neurochemistry.neuromodulators import EndocrineSystem

        if isinstance(module, FeelingSystem):
            self._feelings = module
            print(f'>>> FeelingSystem подключена: {list(f.name for f in module)}')
        elif isinstance(module, EndocrineSystem):
            self._chemistry = module
            print('>>> EndocrineSystem (нейрохимия) подключена')
        else:
            raise TypeError(
                f"Unsupported module type: {type(module).__name__}. "
                f"Expected FeelingSystem or EndocrineSystem.")
        return self

    # -------------------------------------------------------------------------
    # Persistence — directory format (brain separate from world)
    # -------------------------------------------------------------------------

    def load(self, path: str | None = None) -> bool:
        """Load brain weights from the directory save format.

        Returns True on success.
        """
        p = path or self.save_path
        res = self._load_dir(p)
        if res:
            self._base_delays = build_delay_tensor(
                self._plasticity.indices[0],
                self._plasticity.indices[1],
                self._geometry.coordinates,
                device=self._device,
            )
            self.update_myelinated_delays()
        return res

    def update_myelinated_delays(self) -> None:
        """Scale conduction delay dynamically based on synaptic integrity (myelination).

        As synapses are active and consolidate (high integrity), oligodendrocytes
        myelinate the axons, increasing conduction velocity (reducing delay).
        """
        integrity = self._plasticity.integrity_values
        scaled_delays = self._base_delays.float() / (1.0 + 1.5 * integrity)
        self._edge_delays = torch.clamp(torch.round(scaled_delays), min=1.0).to(torch.int16)

    def save(self, path: str | None = None, world = None) -> None:
        """Save brain weights (brain state only — world state not included)."""
        p = path or self.save_path
        self._save_dir(p, world)

    # --- directory save format -------------------------------------------------

    def _save_dir(self, save_dir: str, world = None) -> None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        try:
            idx = self._plasticity.indices.cpu().numpy()
            w   = self._plasticity.weights_values.cpu().numpy()
            ig  = self._plasticity.integrity_values.cpu().numpy()
            # Astrocytic slow EMA — long-term memory stabiliser
            slow_w = (self._astrocytes._slow_weights.cpu().numpy()
                      if self._astrocytes._slow_weights is not None else None)
            avg_act = (self._astrocytes._avg_activity.cpu().numpy()
                       if self._astrocytes._avg_activity is not None else None)
            payload = {
                'row': idx[0], 'col': idx[1],
                'weights': w, 'integrity': ig,
                'num_neurons': np.array(self.num_neurons),
            }
            if slow_w is not None:
                payload['astro_slow'] = slow_w
            if avg_act is not None:
                payload['astro_avg_act'] = avg_act

            # Predictive hierarchy weights
            payload['pred_td_indices'] = self._predictor.td_indices.cpu().numpy()
            payload['pred_td_values'] = self._predictor.td_values.cpu().numpy()
            payload['pred_bu_indices'] = self._predictor.bu_indices.cpu().numpy()
            payload['pred_bu_values'] = self._predictor.bu_values.cpu().numpy()
            payload['pred_td_perm'] = self._predictor.td_perm.cpu().numpy()
            payload['pred_bu_perm'] = self._predictor.bu_perm.cpu().numpy()

            # Language cortex weights
            payload['lang_wernicke_W'] = self._language_cortex.wernicke._W
            payload['lang_wernicke_b'] = self._language_cortex.wernicke._b
            ctx_buf = self._language_cortex.wernicke._context_buf
            if ctx_buf:
                payload['lang_wernicke_context_buf'] = np.stack(ctx_buf)
            payload['lang_broca_W_prod'] = self._language_cortex.broca._W_prod
            payload['lang_broca_W_stack'] = self._language_cortex.broca._W_stack
            payload['lang_broca_W_arcuate'] = self._language_cortex.broca._W_arcuate

            # Cross-modal binder weights
            for key, val in self._cross_modal_binder._W.items():
                payload[f'cmb_W_{key[0]}_{key[1]}'] = val

            # Dialogue ToM weights
            if self._tom.dialogue is not None:
                payload['tom_dialogue_W_mentalise'] = self._tom.dialogue._W_mentalise
                payload['tom_dialogue_W_affect'] = self._tom.dialogue._W_affect
                payload['tom_dialogue_topic'] = self._tom.dialogue.interlocutor_topic
                payload['tom_dialogue_belief'] = self._tom.dialogue.belief_about_other_knowledge
                payload['tom_dialogue_affect'] = self._tom.dialogue.simulated_affect
                hist = list(self._tom.dialogue._history)
                if hist:
                    payload['tom_dialogue_history'] = np.stack(hist)

            # PFC working memory state
            if self._pfc.dialogue_wm is not None:
                payload['pfc_dialogue_wm_context'] = self._pfc.dialogue_wm.context_vector
                payload['pfc_dialogue_wm_drive'] = np.array(self._pfc.dialogue_wm.communicative_drive)
                wm_buf = self._pfc.dialogue_wm._buffer
                if wm_buf:
                    payload['pfc_dialogue_wm_vectors'] = np.stack([e['vector'] for e in wm_buf])
                    payload['pfc_dialogue_wm_strengths'] = np.array([e['strength'] for e in wm_buf])
                    payload['pfc_dialogue_wm_roles'] = np.array([e['role'] for e in wm_buf])

            np.savez_compressed(str(d / 'weights.npz'), **payload)

            meta = {
                'tick':            self.tick,
                'num_neurons':     self.num_neurons,
                'active_limit':    int(self._plasticity.active_limit),
                'wellbeing':       self.wellbeing,
                'mood':            self.mood,
                'rehab_ticks_left': self._rehab_ticks_left,
                'age_ticks':       self._plasticity._age_ticks,
                'habenula_expect': self._habenula._expected_reward,
                'pfc_monitoring':  self._pfc_monitoring_strength,
            }
            (d / 'brain.json').write_text(json.dumps(meta, indent=2))

            # Append metadata log entry (JSONL format)
            try:
                import datetime
                log_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'tick': self.tick,
                    'phase': getattr(world, '_phase', 'unknown') if world is not None else 'unknown',
                    'wellbeing': self.wellbeing,
                    'mood': self.mood
                }
                with open(d / 'metadata.jsonl', 'a', encoding='utf-8') as lf:
                    lf.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception:
                pass

            # Hippocampal subfields — episodic memory + CA3 attractor
            hipp_state = {
                'W_ca3_rec':      self._hipp_subf._W_ca3_rec,
                'stored_count':   np.array(len(self._hipp_subf._stored_patterns)),
            }
            if self._hipp_subf._stored_patterns:
                hipp_state['stored'] = np.stack(self._hipp_subf._stored_patterns)
            np.savez_compressed(str(d / 'hippocampus.npz'), **hipp_state)

            # Entorhinal — learned semantic positions of concepts
            ento_state = {
                'position':     self._entorhinal.position.tolist(),
                'concept_pos':  {str(k): v.tolist() for k, v in
                                 self._entorhinal._concept_pos.items()},
            }
            (d / 'entorhinal.json').write_text(json.dumps(ento_state))

            # Basal Ganglia, Somatic Markers & Endocrine Pavlovian weights (behavior state)
            behavior_state = {
                'direct_weights': self._basal_ganglia.direct_weights,
                'indirect_weights': self._basal_ganglia.indirect_weights,
                'somatic_markers': self._free_will.somatic_markers_pain,
            }
            if self._chemistry is not None and hasattr(self._chemistry, 'sound_to_dopamine_weights'):
                behavior_state['sound_to_dopamine'] = self._chemistry.sound_to_dopamine_weights
                behavior_state['sound_to_fear'] = self._chemistry.sound_to_fear_weights
            np.savez_compressed(str(d / 'behavior.npz'), **behavior_state)

            print(f'>>> Brain saved to {d}/')
        except Exception as e:
            print(f'>>> Error saving brain: {e}')

    def _load_dir(self, save_dir: str) -> bool:
        d = Path(save_dir)
        npz_path  = d / 'weights.npz'
        json_path = d / 'brain.json'
        if not npz_path.exists():
            return False
        try:
            data = np.load(str(npz_path))
            dev  = self._device
            self._plasticity.indices = torch.tensor(
                np.vstack((data['row'], data['col'])), dtype=torch.long, device=dev)
            self._plasticity.weights_values = torch.tensor(
                data['weights'], dtype=torch.float32, device=dev)
            self._plasticity.integrity_values = torch.tensor(
                data['integrity'], dtype=torch.float32, device=dev)
            if 'astro_slow' in data.files:
                self._astrocytes._slow_weights = torch.tensor(
                    data['astro_slow'], dtype=torch.float32, device=dev)
            if 'astro_avg_act' in data.files:
                self._astrocytes._avg_activity = torch.tensor(
                    data['astro_avg_act'], dtype=torch.float32, device=dev)

            # Predictive hierarchy weights
            if 'pred_td_indices' in data.files:
                self._predictor.td_indices = torch.tensor(
                    data['pred_td_indices'], dtype=torch.long, device=dev)
            if 'pred_td_values' in data.files:
                self._predictor.td_values = torch.tensor(
                    data['pred_td_values'], dtype=torch.float32, device=dev)
                self._predictor._W_top = None
            if 'pred_bu_indices' in data.files:
                self._predictor.bu_indices = torch.tensor(
                    data['pred_bu_indices'], dtype=torch.long, device=dev)
            if 'pred_bu_values' in data.files:
                self._predictor.bu_values = torch.tensor(
                    data['pred_bu_values'], dtype=torch.float32, device=dev)
                self._predictor._W_bot = None
            if 'pred_td_perm' in data.files:
                self._predictor.td_perm = torch.tensor(
                    data['pred_td_perm'], dtype=torch.long, device=dev)
            if 'pred_bu_perm' in data.files:
                self._predictor.bu_perm = torch.tensor(
                    data['pred_bu_perm'], dtype=torch.long, device=dev)

            # Language cortex weights
            if 'lang_wernicke_W' in data.files:
                self._language_cortex.wernicke._W = data['lang_wernicke_W'].copy()
            if 'lang_wernicke_b' in data.files:
                self._language_cortex.wernicke._b = data['lang_wernicke_b'].copy()
            if 'lang_wernicke_context_buf' in data.files:
                ctx_buf = data['lang_wernicke_context_buf']
                if ctx_buf.ndim > 1:
                    self._language_cortex.wernicke._context_buf = list(ctx_buf)
                else:
                    self._language_cortex.wernicke._context_buf = []
            if 'lang_broca_W_prod' in data.files:
                self._language_cortex.broca._W_prod = data['lang_broca_W_prod'].copy()
            if 'lang_broca_W_stack' in data.files:
                self._language_cortex.broca._W_stack = data['lang_broca_W_stack'].copy()
            if 'lang_broca_W_arcuate' in data.files:
                self._language_cortex.broca._W_arcuate = data['lang_broca_W_arcuate'].copy()

            # Cross-modal binder weights
            for key in self._cross_modal_binder._W.keys():
                w_key = f'cmb_W_{key[0]}_{key[1]}'
                if w_key in data.files:
                    self._cross_modal_binder._W[key] = data[w_key].copy()

            # Dialogue ToM weights
            if self._tom.dialogue is not None:
                if 'tom_dialogue_W_mentalise' in data.files:
                    self._tom.dialogue._W_mentalise = data['tom_dialogue_W_mentalise'].copy()
                if 'tom_dialogue_W_affect' in data.files:
                    self._tom.dialogue._W_affect = data['tom_dialogue_W_affect'].copy()
                if 'tom_dialogue_topic' in data.files:
                    self._tom.dialogue.interlocutor_topic = data['tom_dialogue_topic'].copy()
                if 'tom_dialogue_belief' in data.files:
                    self._tom.dialogue.belief_about_other_knowledge = data['tom_dialogue_belief'].copy()
                if 'tom_dialogue_affect' in data.files:
                    self._tom.dialogue.simulated_affect = data['tom_dialogue_affect'].copy()
                if 'tom_dialogue_history' in data.files:
                    hist = data['tom_dialogue_history']
                    self._tom.dialogue._history.clear()
                    if hist.ndim > 1:
                        for h in hist:
                            self._tom.dialogue._history.append(h)

            # PFC working memory state
            if self._pfc.dialogue_wm is not None:
                if 'pfc_dialogue_wm_context' in data.files:
                    self._pfc.dialogue_wm.context_vector = data['pfc_dialogue_wm_context'].copy()
                if 'pfc_dialogue_wm_drive' in data.files:
                    self._pfc.dialogue_wm.communicative_drive = float(data['pfc_dialogue_wm_drive'])
                if 'pfc_dialogue_wm_vectors' in data.files:
                    vectors = data['pfc_dialogue_wm_vectors']
                    strengths = data['pfc_dialogue_wm_strengths']
                    roles = data['pfc_dialogue_wm_roles']
                    self._pfc.dialogue_wm._buffer = []
                    if vectors.ndim > 1:
                        for i in range(len(vectors)):
                            self._pfc.dialogue_wm._buffer.append({
                                'vector': vectors[i].copy(),
                                'strength': float(strengths[i]),
                                'role': str(roles[i]),
                            })
            if json_path.exists():
                meta = json.loads(json_path.read_text())
                self.tick      = meta.get('tick', 0)
                self.mood      = meta.get('mood', 'calm')
                self.wellbeing = meta.get('wellbeing', 1.0)
                self._rehab_ticks_left = meta.get('rehab_ticks_left', 0)
                if 'active_limit'    in meta: self._plasticity.active_limit = meta['active_limit']
                if 'age_ticks'       in meta: self._plasticity._age_ticks   = meta['age_ticks']
                if 'habenula_expect' in meta: self._habenula._expected_reward = meta['habenula_expect']
                if 'pfc_monitoring'  in meta: self._pfc_monitoring_strength   = meta['pfc_monitoring']
            # Hippocampal subfields
            hipp_p = d / 'hippocampus.npz'
            if hipp_p.exists():
                hd = np.load(str(hipp_p))
                self._hipp_subf._W_ca3_rec = hd['W_ca3_rec']
                if 'stored' in hd.files:
                    self._hipp_subf._stored_patterns = list(hd['stored'])
                    self._hipp_subf._episode_meta = [{} for _ in self._hipp_subf._stored_patterns]
            # Entorhinal positions
            ento_p = d / 'entorhinal.json'
            if ento_p.exists():
                es = json.loads(ento_p.read_text())
                self._entorhinal.position = np.asarray(es.get('position', [0,0]), dtype=np.float32)
                self._entorhinal._concept_pos = {
                    int(k): np.asarray(v, dtype=np.float32)
                    for k, v in es.get('concept_pos', {}).items()
                }
            # Basal Ganglia, Somatic Markers & Endocrine Pavlovian weights
            behavior_p = d / 'behavior.npz'
            if behavior_p.exists():
                bd = np.load(str(behavior_p), allow_pickle=True)
                if 'direct_weights' in bd:
                    self._basal_ganglia.direct_weights = bd['direct_weights']
                if 'indirect_weights' in bd:
                    self._basal_ganglia.indirect_weights = bd['indirect_weights']
                if 'somatic_markers' in bd:
                    self._free_will.somatic_markers_pain = bd['somatic_markers']
                    self._free_will._prev_markers = np.zeros_like(bd['somatic_markers'])
                if 'sound_to_dopamine' in bd and self._chemistry is not None and hasattr(self._chemistry, 'sound_to_dopamine_weights'):
                    self._chemistry.sound_to_dopamine_weights = bd['sound_to_dopamine']
                if 'sound_to_fear' in bd and self._chemistry is not None and hasattr(self._chemistry, 'sound_to_fear_weights'):
                    self._chemistry.sound_to_fear_weights = bd['sound_to_fear']
            print(f'>>> Brain loaded from {d}/ (tick {self.tick})')
            return True
        except Exception as e:
            print(f'>>> Error loading brain: {e}')
            return False

    def _setup_runtime_io(self, headless: bool, layout):
        from mindai.environment.hearing_system import Cochlea

        if headless:
            return None
        return Cochlea(num_bands=layout.size('audio'))

    def _shutdown_runtime(self, ear, world, sp_path: str) -> None:
        if ear is not None and hasattr(ear, 'stream') and ear.stream is not None:
            ear.stream.stop()
        if world.is_alive():
            self.save(sp_path, world=world)

    def _prepare_awake_inputs(
        self,
        world,
        layout,
        chem,
        ear,
        raw_buffer,
        sl_vision,
        sl_pain,
        sl_hunger,
        sl_audio,
        sl_voc,
        sz_audio,
    ) -> dict:
        world_signals = world.get_homeostatic_signals()

        if self._feelings is not None:
            self._feelings.update(world_signals)
            self.wellbeing = self._feelings.wellbeing()
        else:
            self.wellbeing = 1.0 - float(np.mean(list(world_signals.values()) or [0.0]))

        h_ratio = 1.0 - world_signals.get('hunger', 0.0)
        w_ratio = 1.0 - world_signals.get('thirst', 0.0)
        self._last_h_ratio = h_ratio
        self._last_w_ratio = w_ratio

        raw_pain_for_chem = world_signals.get('pain', 0.0)
        if self._feelings is not None and 'pain' in self._feelings:
            raw_pain_for_chem = self._feelings['pain'].raw
        p_sig = chem.effective_pain_signal

        raw = raw_buffer
        raw[:] = 0.0

        retina_data = world.get_sensory_retina(self.num_neurons)
        if isinstance(retina_data, dict):
            for channel, data in retina_data.items():
                if layout.has(channel):
                    slice_dest = layout.slice(channel)
                    slice_len = layout.size(channel)
                    raw[slice_dest] = np.pad(data, (0, max(0, slice_len - len(data))))[:slice_len]
        else:
            raw[:len(retina_data)] = retina_data

        vis_data = raw[sl_vision] if sl_vision is not None else np.zeros(0)
        bio_motion_score = self._motion_detector.update(vis_data)

        pfc_goal = self._pfc.formulate_goal(
            energy=h_ratio,
            water=w_ratio,
            base_resource=1.0,
        )
        raw += pfc_goal * 0.3

        if self._feelings is not None:
            if 'pain' in self._feelings and sl_pain is not None:
                raw[sl_pain] = max(p_sig, self._feelings['pain'].sensation)
            for feel in self._feelings:
                if feel.channel == 'pain':
                    continue
                if feel.channel == 'hunger':
                    sig = max(feel.sensation, chem.effective_hunger_signal)
                    if sl_hunger is not None:
                        raw[sl_hunger] = min(1.0, sig)
                elif layout.has(feel.channel):
                    raw[layout.slice(feel.channel)] = feel.sensation
        else:
            for channel, deficit in world_signals.items():
                if channel == 'pain' and sl_pain is not None:
                    raw[sl_pain] = max(p_sig, float(deficit))
                elif layout.has(channel):
                    raw[layout.slice(channel)] = float(np.clip(deficit, 0, 1))

        if ear is not None:
            mic_audio = ear.get_auditory_nerve_signal()
        else:
            mic_audio = np.zeros(sz_audio, dtype=np.float32)

        if sl_audio is not None:
            def _fit(x, n):
                return np.pad(x, (0, max(0, n - len(x))))[:n]

            world_sound = _fit(world.pop_world_sound(), sz_audio)
            echo_read_idx = (self._voc_echo_idx + 1) % 3
            vocal_echo = _fit(self._voc_echo_buf[echo_read_idx] * 0.4, sz_audio)
            voc = world.last_agent_vocalization if sl_voc is not None else np.zeros(0)
            voc_part = _fit(voc, sz_audio) if len(voc) else 0
            raw[sl_audio] = _fit(mic_audio, sz_audio) + world_sound + vocal_echo + voc_part

        dmn_replay = getattr(self, '_dmn_replay_prev', None)
        if dmn_replay is not None and len(dmn_replay):
            n = min(len(dmn_replay), len(raw))
            raw[:n] += dmn_replay[:n] * 0.15

        insula_out = self._insula.update(
            pain=world_signals.get('pain', 0.0),
            hunger=world_signals.get('hunger', 0.0),
            thirst=world_signals.get('thirst', 0.0),
            arousal=float(np.mean(raw)),
        )

        amyg_out = self._amygdala.update(raw, raw_pain_for_chem, chem.dopamine)
        if self._chemistry is not None:
            threat_na = amyg_out['threat_level'] * (1.0 - chem.serotonin * 0.5)
            self._chemistry.noradrenaline = float(np.clip(
                chem.noradrenaline + threat_na * 0.05, 0.1, 1.0))
            self._chemistry.dopamine = float(np.clip(
                chem.dopamine - amyg_out['da_suppression'] * 0.02, 0.1, 1.0))

        return {
            'world_signals': world_signals,
            'h_ratio': h_ratio,
            'w_ratio': w_ratio,
            'raw_pain_for_chem': raw_pain_for_chem,
            'p_sig': p_sig,
            'raw': raw,
            'mic_audio': mic_audio,
            'bio_motion_score': bio_motion_score,
            'insula_out': insula_out,
            'amyg_out': amyg_out,
        }

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def run(
        self,
        world,
        headless:  bool = False,
        save_path: str | None = None,
        max_ticks: int | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        """Run the simulation loop.

        Parameters
        ----------
        world:
            World/environment object supplying the AgentWorld-compatible IO hooks.
        headless:
            Skip PyGame rendering and microphone input.
        save_path:
            Override ``self.save_path`` for this run.
        max_ticks:
            Stop after N ticks (None = run until quit or death).
        checkpoint_interval:
            Save the brain every N ticks (None = no periodic saves).
        """
        sp_path = save_path or self.save_path
        layout  = self._layout

        chem = self._chemistry if self._chemistry is not None else _NULL_CHEM
        ear = self._setup_runtime_io(headless, layout)

        if layout.has('vocalization'):
            world.last_agent_vocalization = np.zeros(layout.size('vocalization'))
        if not hasattr(world, 'isolation_ticks'):
            world.isolation_ticks = 0

        qualia_shape = [0.33, 0.33, 0.33]

        # Pre-cache layout slices — avoids dict lookup + slice() call each tick
        _sl = lambda n: layout.slice(n) if layout.has(n) else None
        _sl_vision, _sl_pain, _sl_hunger = _sl('vision'), _sl('pain'), _sl('hunger')
        _sl_audio,  _sl_motor, _sl_voc   = _sl('audio'),  _sl('motor'), _sl('vocalization')
        _sz_audio  = layout.size('audio')        if layout.has('audio')        else 32
        _sz_voc    = layout.size('vocalization') if layout.has('vocalization') else 0

        # Reusable numpy sensory array — avoids np.zeros() allocation every tick
        _raw_np = np.zeros(self.num_neurons, dtype=np.float32)

        print('>>> Starting AGI...')
        try:
            while True:
                self.tick += 1
                if max_ticks and self.tick > max_ticks:
                    break

                if self._sleep.is_sleeping:
                    # Spawn background sleep consolidation thread if not already running
                    if getattr(self, '_sleep_thread', None) is None or not self._sleep_thread.is_alive():
                        import threading
                        self._sleep_thread = threading.Thread(
                            target=self._run_sleep_cycle,
                            args=(world, chem, sp_path),
                            daemon=True
                        )
                        self._sleep_thread.start()

                    time.sleep(0.016)  # Yield CPU in main thread to keep UI/Pygame responsive (60 FPS)
                    continue

                # === AWAKE PATH ===============================================
                else:
                    awake = self._prepare_awake_inputs(
                        world=world,
                        layout=layout,
                        chem=chem,
                        ear=ear,
                        raw_buffer=_raw_np,
                        sl_vision=_sl_vision,
                        sl_pain=_sl_pain,
                        sl_hunger=_sl_hunger,
                        sl_audio=_sl_audio,
                        sl_voc=_sl_voc,
                        sz_audio=_sz_audio,
                    )
                    world_signals = awake['world_signals']
                    h_ratio = awake['h_ratio']
                    w_ratio = awake['w_ratio']
                    raw_pain_for_chem = awake['raw_pain_for_chem']
                    p_sig = awake['p_sig']
                    raw = awake['raw']
                    mic_audio = awake['mic_audio']
                    _bio_motion_score = awake['bio_motion_score']
                    insula_out = awake['insula_out']
                    amyg_out = awake['amyg_out']

                    # ----------------------------------------------------------
                    # 3. Neural computation
                    # ----------------------------------------------------------
                    # Transfer raw to GPU via pinned memory (zero-copy DMA on CUDA).
                    # np.clip on CPU replaced by torch.clamp on GPU — one less array pass.
                    if self._raw_pinned is not None:
                        self._raw_pinned.numpy()[:] = raw          # write into pinned RAM
                        raw_t = self._raw_pinned.to(                # non-blocking DMA
                            self._device, non_blocking=True)
                    else:
                        raw_t = torch.from_numpy(raw)              # CPU: zero-copy view

                    # Canonical microcircuit bias applied once at init (CorticalLayers).
                    # No global layer routing here — layers exist within columns, not as
                    # global bands across the network (Douglas & Martin 1991).

                    # Axonal delays: add PSPs that arrived this tick from past spikes
                    delayed_psp = self._delay_queue.dequeue()
                    # Schedule current activity into future slots (STP-scaled weights)
                    eff_weights = self._plasticity.get_stp_scaled_weights()
                    # Rebuild edge-delay tensor if synaptogenesis/pruning changed topology
                    n_edges = self._plasticity.indices.shape[1]
                    if not hasattr(self, '_base_delays') or self._base_delays.shape[0] != n_edges:
                        self._base_delays = build_delay_tensor(
                            self._plasticity.indices[0],
                            self._plasticity.indices[1],
                            self._geometry.coordinates,
                            device=self._device,
                        )
                        self.update_myelinated_delays()
                    elif self.tick % 20 == 0:
                        # Periodically update myelination based on active plasticity consolidation (integrity)
                        self.update_myelinated_delays()
                    self._delay_queue.enqueue(
                        self._activity,
                        self._plasticity.indices[0],
                        self._plasticity.indices[1],
                        eff_weights,
                        self._edge_delays,
                    )
                    # Step STP state for next tick
                    self._plasticity.step_stp(self._activity)

                    # Recurrent drive = delayed PSPs (biologically correct);
                    # instantaneous sparse.mm kept as small direct-coupling leak (0.05×)
                    # because axonal delays are capped at 20 ticks — very long-range
                    # connections beyond that still need some path.
                    brain_w   = self._plasticity.get_sparse_weights()
                    recurrent = delayed_psp + torch.sparse.mm(
                        brain_w, self._activity.unsqueeze(1)).squeeze(1) * 0.05
                    # clamp on GPU — no CPU round-trip for clip
                    combined  = torch.clamp(raw_t + recurrent * 0.1, 0.0, 1.0)

                    surprise_t, fep_state = self._predictor.process_inference_step(
                        combined, self._activity, plasticity_rate=0.5)
                    self.surprise  = float(surprise_t) if hasattr(surprise_t, '__float__') else surprise_t.item()
                    self._activity = torch.clamp(
                        self._time.create_conscious_now(combined, fep_state), 0.0, 1.0)

                    # Refractory period + spike-frequency adaptation (Hodgkin & Huxley 1952)
                    self._activity = self._plasticity.apply_neural_dynamics(
                        self._activity, acetylcholine=getattr(chem, 'acetylcholine', 0.5))

                    # Cortical lateral inhibition — GABAergic basket cells (Buzsáki 2004)
                    # Applied after refractory/adaptation, before thalamic gating.
                    # ACh modulates competition width (Hasselmo & McGaughy 2004).
                    self._activity = self._plasticity.apply_lateral_inhibition(
                        self._activity,
                        target_sparsity=0.05,
                        acetylcholine=getattr(chem, 'acetylcholine', 0.5),
                    )

                    salient = self._thalamus.filter_attention(
                        self._activity, chem.noradrenaline, chem.boredom)
                    if salient.any():
                        self._activity = self._workspace.broadcast_via_synchrony(
                            salient, self._activity)
                        if (self.surprise > 5.0 or p_sig > 0.2
                                or np.sum(mic_audio) > 1.0):
                            raw_salience   = chem.dopamine - p_sig
                            gated_salience = raw_salience * chem.hippocampal_salience_gate
                            self._hippocampus.encode_episode(
                                self._activity.cpu().numpy(), gated_salience)

                    act_cpu = self._activity.cpu().numpy()

                    # Superior Colliculus — gaze driven by internal brain state.
                    # SC integrates motion salience, surprise, threat, and neuromodulators.
                    # No scripted rule: the agent looks where desire/interest/fear points.
                    # IOR emerges from burst-neuron refractory, not from external code.
                    if self._sc is not None and _sl_vision is not None:
                        _vis    = act_cpu[_sl_vision]
                        _motion = _vis[4::5]   # motion  channel (temporal change)
                        _luma   = _vis[3::5]   # luma    channel (includes top-down)
                        # 5-HT suppresses CeA→SC threat projection (Gross & Canteras 2012)
                        _threat_sc = self._amygdala.threat_level * (1.0 - chem.serotonin * 0.5)
                        _fx, _fy, _sacc = self._sc.update(
                            visual_motion  = _motion,
                            visual_luma    = _luma,
                            surprise       = self.surprise,
                            threat         = _threat_sc,
                            dopamine       = chem.dopamine,
                            noradrenaline  = chem.noradrenaline,
                            acetylcholine  = chem.acetylcholine,
                            goal_drive     = self._pfc.goal_persistence,
                        )
                        if hasattr(world, 'receive_gaze'):
                            world.receive_gaze(_fx, _fy)

                    # Cortical areas — update area-wise activity map (Zeki 1978)
                    self._areas.update(act_cpu)

                    # Entorhinal grid cells — semantic position drifts with the
                    # current input token (Hafting 2005). Hippocampus DG/CA3/CA1
                    # then performs pattern separation + completion (Marr 1971).
                    _ento_input = act_cpu[:self._entorhinal.total_cells]
                    self._entorhinal.advance(int(self._activity.argmax().item()))
                    _grid = self._entorhinal.get_grid_activity()
                    # Pattern-separated hippocampal encoding of the current state
                    _hipp_out = self._hipp_subf.encode(
                        np.concatenate([_grid, act_cpu[:self._hipp_subf.input_size - len(_grid)]])
                            if len(_grid) < self._hipp_subf.input_size else _grid[:self._hipp_subf.input_size]
                    )
                    # Novelty (CA1 mismatch) → mesocortical DA boost (Lisman & Grace 2005)
                    if self._chemistry is not None and _hipp_out['novelty'] > 0.7:
                        self._chemistry.dopamine_mesocortical = float(np.clip(
                            chem.dopamine_mesocortical + 0.01, 0.1, 1.0))
                    # Store every ~50 ticks so CA3 builds an associative trace
                    if self.tick % 50 == 0:
                        self._hipp_subf.store({'tick': self.tick, 'mood': self.mood})

                    # Olfactory bulb — direct-to-amygdala route (Buck & Axel 1991).
                    # No real chemoreceptors here; we treat very low-frequency
                    # auditory bins as a coarse "scent of valence" surrogate so
                    # the pathway is exercised. Output adds to amygdala arousal.
                    if _sl_audio is not None:
                        _scent = act_cpu[_sl_audio][:self._olfactory.num_glomeruli]
                        if _scent.size < self._olfactory.num_glomeruli:
                            _scent = np.pad(_scent, (0, self._olfactory.num_glomeruli - _scent.size))
                        self._olfactory.update(_scent * 0.3)

                    # Visuospatial sketchpad — spatial working memory (Baddeley 1986)
                    # retina_data from step 2 above — vision channel
                    self._visuospatial.update(raw[_sl_vision] if _sl_vision is not None else np.zeros(1))

                    # DMN — self-referential / autobiographical replay (Raichle 2001)
                    dmn_out = self._dmn.update(
                        external_arousal=float(np.mean(act_cpu)),
                        episodic_memory=self._hippocampus.episodic_memory,
                        wellbeing=self.wellbeing,
                    )
                    self._is_daydreaming = dmn_out['activation'] > 0.4
                    self._dmn_replay_prev = dmn_out['replay_pattern']

                    # LanguageCortex + CrossModalBinder + dlPFC WorkingMemory
                    # Runs every tick so the system continuously processes the
                    # current sensory state as a "token" in semantic space.
                    # When text arrives the SDR gives a proper sparse code;
                    # otherwise the dominant neural activity vector is used.
                    # ----------------------------------------------------------

                    # --- dlPFC: decay WM + retrieve context ---
                    # Goldman-Rakic (1995): each tick corresponds to ~100ms;
                    # WM decays exponentially without active rehearsal.
                    _dlpfc_wm = self._pfc.dialogue_wm
                    if _dlpfc_wm is not None:
                        _dlpfc_wm.decay()
                        # Retrieve context cued by current neural state
                        _dlpfc_ctx = _dlpfc_wm.retrieve(
                            self._language_cortex.semantic_vector)
                        # Active rehearsal: refresh item closest to current state
                        _dlpfc_wm.rehearse(self._language_cortex.semantic_vector)
                    else:
                        _dlpfc_ctx = None

                    _text_tok = world.get_text_token() if hasattr(world, 'get_text_token') else None
                    if _text_tok is not None:
                        # Text input available — encode as SDR
                        _tok_arr = np.array([_text_tok], dtype=np.float32)
                        _tok_sdr = self._sdr_encoder.encode(_tok_arr)
                    else:
                        # No text: use top-K of current neural activity as implicit token
                        _tok_sdr = self._sdr_encoder.encode(act_cpu)

                    _grid_act = self._entorhinal.get_grid_activity()

                    # Sentence boundary: triggered by long silence or punctuation
                    _sentence_end = (
                        getattr(world, 'is_sentence_boundary', False) or
                        (self.tick % 50 == 0)  # fallback: every 5 s at 10 Hz
                    )

                    # --- LanguageCortex: process with dlPFC back-projection ---
                    # Petrides & Pandya (1988): BA46/9 → BA22 top-down connection
                    # injects dialogue context into Wernicke's semantic computation.
                    _lang_out = self._language_cortex.process(
                        token_embedding   = _tok_sdr,
                        grid_activity     = _grid_act,
                        target_token      = _text_tok,
                        plasticity_rate   = getattr(chem, 'acetylcholine', 0.5),
                        sentence_boundary = _sentence_end,
                        dlpfc_context     = _dlpfc_ctx,
                    )

                    # Store agent's current semantic state into dlPFC WM
                    # (every sentence boundary or every ~5 s)
                    if _dlpfc_wm is not None and _sentence_end:
                        _dlpfc_wm.store(_lang_out['semantic_vector'], role='self')

                    # Cross-modal binding — fire when multiple modalities present
                    _intero_vec = np.array([
                        world_signals.get('pain',   0.0),
                        world_signals.get('hunger', 0.0),
                        world_signals.get('thirst', 0.0),
                        float(np.mean(act_cpu)),
                    ], dtype=np.float32)
                    _vis_vec  = act_cpu[_sl_vision] if _sl_vision is not None else np.zeros(self._lang_vision_dim)
                    _aud_vec  = act_cpu[_sl_audio]  if _sl_audio  is not None else np.zeros(self._lang_audio_dim)
                    _bind_recall = self._cross_modal_binder.update({
                        'text':         _tok_sdr[:self._binder_text_dim],
                        'audio':        _aud_vec,
                        'vision':       _vis_vec,
                        'interoception': _intero_vec,
                    })
                    # Recalled cross-modal signal reinforces the text column
                    _text_recall = _bind_recall.get('text', None)
                    if _text_recall is not None and _text_recall.any():
                        # Zero-pad recalled text back to full SDR dimension
                        _text_recall_full = np.zeros_like(_tok_sdr)
                        _text_recall_full[:len(_text_recall)] = _text_recall
                        # Blend recalled multimodal signal back into SDR encoder
                        self._sdr_encoder.update_reconstruction(_tok_sdr, _text_recall_full)

                    # Dialogue ToM update — called when an interlocutor speaks
                    _interlocutor_text = getattr(world, 'last_interlocutor_token', None)
                    _knowledge_gap = 0.0
                    if (_interlocutor_text is not None
                            and self._tom.dialogue is not None):
                        _inter_sdr = self._sdr_encoder.encode(
                            np.array([_interlocutor_text], dtype=np.float32))
                        _inter_sem = self._language_cortex.wernicke.process(
                            token_embedding = _inter_sdr,
                            grid_activity   = _grid_act,
                            plasticity_rate = 0.0,   # observe only, no self-update
                            dlpfc_context   = _dlpfc_ctx,
                        )
                        # Store interlocutor's utterance into dlPFC WM
                        if _dlpfc_wm is not None:
                            _dlpfc_wm.store(_inter_sem, role='other')

                        # Build agent feeling vector for simulation theory
                        _agent_feel = np.array([
                            getattr(chem, 'dopamine', 0.5) - 0.5,   # valence proxy
                            getattr(chem, 'noradrenaline', 0.5),     # arousal proxy
                        ], dtype=np.float32)
                        _tom_out = self._tom.dialogue.update_from_utterance(
                            observed_semantic = _inter_sem,
                            agent_semantic    = _lang_out['semantic_vector'],
                            agent_feeling     = _agent_feel,
                            plasticity_rate   = getattr(chem, 'acetylcholine', 0.5),
                        )

                        _knowledge_gap = _tom_out.get('knowledge_gap', 0.0)

                    # --- Communicative intent (ACC-gated dlPFC drive) ---
                    # Indefrey & Levelt (2004): speech triggered when SMA readiness
                    # potential exceeds threshold.  Here: communicative_drive >
                    # _COMM_DRIVE_THRESHOLD AND agent not in homeostatic crisis.
                    _just_spoke = False
                    if _dlpfc_wm is not None:
                        _comm_drive = _dlpfc_wm.update_communicative_drive(
                            surprise      = self.surprise,
                            knowledge_gap = _knowledge_gap,
                            dopamine_meso = getattr(chem, 'dopamine_mesocortical', 0.5),
                            just_spoke    = _just_spoke,
                        )
                        # Trigger generation when drive is high AND agent is not
                        # distracted by homeostatic crisis (goal_persistence low)
                        _can_speak = (
                            _comm_drive > _COMM_DRIVE_THRESHOLD and
                            self._pfc.goal_persistence < 0.8 and
                            not self._pag.is_immobile   # PAG freeze blocks speech
                        )
                        if _can_speak:
                            _utterance_tokens = self._language_cortex.generate_utterance(
                                grid_activity = _grid_act,
                                sdr_encoder   = self._sdr_encoder,
                                dlpfc_context = _dlpfc_ctx,
                            )
                            if _utterance_tokens:
                                # Pass tokens to world for rendering/output
                                if hasattr(world, 'receive_agent_tokens'):
                                    world.receive_agent_tokens(_utterance_tokens)
                                # Store generated utterance back into WM
                                if _dlpfc_wm is not None:
                                    _dlpfc_wm.store(
                                        _lang_out['semantic_vector'], role='self')
                                # Reset drive: information has been transmitted
                                _dlpfc_wm.update_communicative_drive(
                                    surprise=0.0, knowledge_gap=0.0,
                                    dopamine_meso=0.5, just_spoke=True)

                    # Neural complexity (every 100 ticks)
                    # LZ76 complexity + spectral radius via GPU power iteration.
                    # Replaces scipy CSR build which spiked memory to 2+ GB.
                    if self.tick % 100 == 0:
                        _w_pi = self._plasticity.get_sparse_weights()
                        _v_pi = self._activity.clone()
                        _n_pi = _v_pi.norm()
                        _spectral_r = 0.0
                        if _n_pi > 1e-12:
                            _v_pi = _v_pi / _n_pi
                            for _ in range(15):
                                _v_pi = torch.sparse.mm(
                                    _w_pi, _v_pi.unsqueeze(1)).squeeze(1)
                                _n_pi = _v_pi.norm()
                                if _n_pi < 1e-12:
                                    break
                                _v_pi = _v_pi / _n_pi
                            _spectral_r = float(
                                torch.sparse.mm(_w_pi, _v_pi.unsqueeze(1)).squeeze(1).norm())
                        qualia_shape = self._nc.calculate(act_cpu, spectral_radius=_spectral_r)
                        # Spectral radius proxy → ignition bias (Beggs & Plenz 2003):
                        # near-critical networks (ρ≈1) show maximal dynamic range.
                        # High complexity → lower ignition threshold.
                        sr = qualia_shape[1]   # spectral_radius_proxy
                        self._workspace._ignition_bias = float(
                            np.clip(0.25 - sr * 0.2, 0.05, 0.25))

                    # ----------------------------------------------------------
                    # 4. Motor output
                    # ----------------------------------------------------------
                    motor_signals = act_cpu[_sl_motor]
                    # PAG freeze: vlPAG inhibits motor output during freeze
                    if self._pag.is_immobile:
                        motor_signals = motor_signals * 0.1
                    if _sl_voc is not None:
                        vocal_cords = act_cpu[_sl_voc]
                        if np.sum(vocal_cords) > 3.0:
                            world.receive_vocalization(vocal_cords.copy())
                            world.add_sound(
                                getattr(world, 'agent_pos', [0, 0]),
                                vocal_cords.copy() * 0.5,
                            )
                            # Enqueue echo into phonological loop ring buffer.
                            # Written at current slot; read 2 ticks later (above).
                            # Amplitude 0.4× — echo is quieter than source.
                            self._voc_echo_buf[self._voc_echo_idx][:] = vocal_cords
                        else:
                            world.receive_vocalization(np.zeros(_sz_voc))
                            self._voc_echo_buf[self._voc_echo_idx][:] = 0.0
                        self._voc_echo_idx = (self._voc_echo_idx + 1) % 3

                    spasm_intensity = float(np.mean(motor_signals))

                    motor_pot = self._basal_ganglia.map_to_action_potentials(motor_signals)

                    # ACC conflict monitoring — before action selection
                    # Computes conflict across competing action potentials
                    _pot_exp = np.exp(motor_pot - motor_pot.max())
                    _pot_prob = _pot_exp / (_pot_exp.sum() + 1e-9)
                    acc_out = self._acc.update(
                        action_probs=_pot_prob,
                        chosen_action=self._free_will.decision_queue[-1]['action']
                            if self._free_will.decision_queue else None,
                        prediction_error=self.surprise,
                        pain_signal=self._amygdala.threat_level,
                    )
                    # High conflict → STN hyperdirect brake (Aron & Poldrack 2006)
                    if acc_out['conflict'] > 0.7:
                        self._basal_ganglia.hyperdirect_brake(acc_out['conflict'])
                    # ACC control request → boost PFC goal persistence
                    if acc_out['control_request'] > 0.5:
                        self._pfc.goal_persistence = min(
                            1.0, self._pfc.goal_persistence + 0.05)

                    # Cerebellum — only meaningful with real proprioceptive reafference
                    if hasattr(world, 'get_proprioception'):
                        cereb_out = self._cerebellum.update(
                            motor_command=motor_signals,
                            actual_reafference=world.get_proprioception(),
                        )
                        if cereb_out['prediction_error'] > 0.1:
                            _corr = cereb_out['correction']
                            motor_pot = motor_pot + np.dot(_corr, self._basal_ganglia.direct_weights)

                    self._free_will.unconscious_decision_making(
                        motor_pot, chem.noradrenaline)
                    final_action = self._free_will.conscious_veto_and_awareness()

                    # ----------------------------------------------------------
                    # 5. World interaction
                    # ----------------------------------------------------------
                    results = {'energy': 0.0, 'water': 0.0, 'stress': 0.0}
                    # If the world can receive the raw motor pattern, pass it
                    # through before execute_action so it can decode its own
                    # motor representation rather than only a BG action index.
                    if hasattr(world, 'receive_motor_pattern'):
                        world.receive_motor_pattern(motor_signals)

                    num_actions = self._basal_ganglia.num_actions
                    if (final_action is not None
                            and final_action < num_actions
                            and not self._is_daydreaming):
                        results = world.execute_action(final_action)
                        energy_gained = results.get('energy', 0.0)
                        stress        = results.get('stress',  0.0)
                        if energy_gained > 0 or results.get('water', 0.0) > 0:
                            chem.trigger_endorphin_rush()
                        # Somatic marker uses felt_distress (substance-P-sensitised
                        # interoceptive signal) not raw world stress — this is the
                        # "somatic" part: what the body felt, not what happened (Damasio 1996)
                        felt_distress = max(p_sig, min(1.0, stress / 100.0))
                        if felt_distress > 0.2:
                            self._free_will.update_somatic_markers(
                                final_action, felt_distress)
                        # Habenula: anti-reward signal on omitted reward
                        # (Matsumoto & Hikosaka 2007). Reward proxy = energy gain.
                        anti_reward = self._habenula.update(
                            actual_reward=energy_gained)
                        if anti_reward > 0.1 and self._chemistry is not None:
                            # LHb → RMTg → VTA: suppress mesolimbic DA
                            self._chemistry.dopamine = float(np.clip(
                                chem.dopamine - anti_reward * 0.05, 0.05, 1.0))
                        # BasalGanglia: three-factor Hebbian — pain acts through DA
                        self._basal_ganglia.reinforce_learning(
                            final_action,
                            dopamine=chem.dopamine,
                        )

                    # PAG defensive mode (fight/flight/freeze) — read by motor
                    # gate during this and next ticks (Bandler & Shipley 1994)
                    self._pag.update(
                        threat=self._amygdala.threat_level,
                        dopamine=chem.dopamine,
                    )
                    # vlPAG opioid analgesia during freeze
                    if self._pag.opioid_analgesia > 0 and self._chemistry is not None:
                        self._chemistry.endorphins = max(
                            chem.endorphins, self._pag.opioid_analgesia)

                    # ----------------------------------------------------------
                    # 6. Neuromodulation (optional — skipped if no EndocrineSystem)
                    # ----------------------------------------------------------
                    if self._chemistry is not None:
                        self._chemistry.update_state(
                            global_arousal=float(np.mean(act_cpu)),
                            layer23_error_spikes=self.surprise,
                            raw_pain_signal=raw_pain_for_chem,
                            energy_ratio=h_ratio,
                            water_ratio=w_ratio,
                            auditory_spikes=mic_audio,
                            energy_gained=max(0.0, results.get('energy', 0.0)),
                            isolation_ticks=getattr(world, 'isolation_ticks', 0),
                        )
                        self.mood = chem.derive_mood()
                        # Mood → plasticity modulation (Castrén 2005; Bhagya 2017)
                        if self.mood == 'depression':
                            chem.noradrenaline = max(0.0, chem.noradrenaline - 0.05)
                            chem.acetylcholine = max(0.05, chem.acetylcholine * 0.97)
                        elif self.mood == 'anxiety':
                            chem.noradrenaline = min(1.0, chem.noradrenaline + 0.05)
                        if hasattr(self._plasticity, 'apply_cortisol_damage'):
                            self._plasticity.apply_cortisol_damage(chem.cortisol)

                    self._ego.evaluate_self(self.wellbeing, raw_pain_for_chem)

                    # Passive somatic marker decay every tick (Milad 2006)
                    self._free_will.decay_markers_passive()

                    # ----------------------------------------------------------
                    # 7. Death check
                    # ----------------------------------------------------------
                    if not world.is_alive():
                        print('\n>>> Organism death.')
                        self._delete_save(sp_path)
                        break

                    # ----------------------------------------------------------
                    # 8. Plasticity & learning
                    # ----------------------------------------------------------
                    # --- Rehabilitation Mode ---
                    rehab_active = (self._rehab_ticks_left > 0)
                    if rehab_active:
                        self._rehab_ticks_left -= 1
                        chem.acetylcholine = max(1.0, chem.acetylcholine)
                        chem.dopamine = max(1.0, chem.dopamine)
                        chem.dopamine_mesocortical = max(1.0, chem.dopamine_mesocortical)
                        if self.tick % 50 == 0:
                            print(f'>>> [REHAB] Remaining rehabilitation time: {self._rehab_ticks_left} ticks.')

                    pain_suppression = max(0.0, 1.0 - raw_pain_for_chem)
                    # Mood gate: depression suppresses LTP rate (Castrén 2005)
                    mood_ltp_gate = 0.5 if self.mood == 'depression' else 1.0
                    plasticity_rate  = (chem.get_plasticity_multiplier()
                                        * pain_suppression * mood_ltp_gate)
                    self._plasticity.apply_stdp_learning(
                        self._activity, plasticity_rate,
                        acetylcholine=getattr(chem, 'acetylcholine', 0.5),
                        dopamine=getattr(chem, 'dopamine', 0.5))

                    # Mirror neuron STDP gate — three-factor amplification:
                    #   1. mirror_neuron_amplifier: oxytocin + NA inverted-U + insula/amygdala
                    #      emotional context (substance_p × (1 − anandamide))
                    #   2. _bio_motion_score: MT+/V5 gate — only amplify when visual input
                    #      looks like biological motion (not static background or camera pan)
                    #   3. Motor activity: mirror neurons only re-fire if motor cortex
                    #      is simultaneously active (Rizzolatti 1996 — observation alone
                    #      is insufficient; the motor program must be primed)
                    # When all three are present: extra STDP pass on mirror_neurons slice
                    # with boosted rate — strengthens V→M path (Iacoboni 1999).
                    if (layout.has('mirror_neurons')
                            and _bio_motion_score > 0.2
                            and chem.mirror_neuron_amplifier > 0.7):
                        _sl_mirror = layout.slice('mirror_neurons')
                        _sl_motor_loc = _sl_motor  # motor activity = motor priming
                        motor_active = float(self._activity[_sl_motor_loc].mean().item()) > 0.1
                        if motor_active:
                            # Isolate mirror zone activity and run an extra STDP pass
                            # with amplified rate — purely Hebbian, no reward signal
                            mirror_rate = (plasticity_rate
                                           * chem.mirror_neuron_amplifier
                                           * _bio_motion_score)
                            mirror_activity = self._activity.clone()
                            # Zero out everything outside mirror zone so STDP only
                            # updates synapses involving these neurons
                            mask = torch.zeros(self.num_neurons,
                                               dtype=torch.bool, device=self._device)
                            mask[_sl_mirror] = True
                            mirror_activity[~mask] = 0.0
                            self._plasticity.apply_stdp_learning(
                                mirror_activity, mirror_rate,
                                acetylcholine=getattr(chem, 'acetylcholine', 0.5),
                                dopamine=getattr(chem, 'dopamine', 0.5))

                    # Astrocytes: slow weight-stability layer (Fusi 2005 cascade)
                    # Pulls inactive synapses toward their slow EMA — preserves
                    # long-term memory without blocking fast STDP (Volterra 2005)
                    rehab_loops = 5 if rehab_active else 1
                    for _ in range(rehab_loops):
                        self._plasticity.synaptogenesis_and_pruning(
                            self._activity, self.wellbeing * 3000)
                    if self.tick % 100 == 0:
                        self._plasticity.maintain_homeostasis()
                    self._plasticity.weights_values = self._astrocytes.step(
                        self._plasticity.weights_values,
                        self._plasticity.indices[0],
                        self._plasticity.indices[1],
                        self._activity,
                    )

                    # Neurogenesis — triggered by surprise (Eriksson 1998; Bhagya 2011)
                    # High prediction error → epistemic hunger → hippocampal neurogenesis
                    self._plasticity.trigger_neurogenesis(self.surprise)

                    # PFC self-monitoring accumulation — lucid dream substrate
                    # (Stumbrys 2012; Erlacher 2008)
                    # Grows when: surprise is high (reality feels anomalous) AND
                    # PFC goal vector is active (metacognitive engagement).
                    # Any waking habit that combines these two will strengthen it —
                    # the brain can discover its own "techniques" through STDP.
                    pfc_active   = float(self._pfc.goal_persistence) > 0.3
                    high_surprise = self.surprise > 3.0
                    if pfc_active and high_surprise:
                        self._pfc_monitoring_strength = min(
                            1.0, self._pfc_monitoring_strength + 0.0002)
                    # Slow decay τ ≈ 5000 ticks
                    self._pfc_monitoring_strength *= 0.9998

                    # Sleep trigger
                    self._clock.update_clock(
                        energy_spent=(1.0 + spasm_intensity * 5.0) * self._clock_energy_scale)
                    if not self._clock.is_awake and not self._sleep.is_sleeping:
                        self._sleep.is_sleeping = True
                        self._sleep.begin_sleep(self._hippocampus)

                    # Periodic checkpoint check
                    if checkpoint_interval is not None and self.tick % checkpoint_interval == 0 and self.tick > 0:
                        print(f'\n>>> [CHECKPOINT] Auto-saving at tick {self.tick:,}...')
                        self.save(sp_path, world=world)

        except KeyboardInterrupt:
            print('\n>>> Emergency interruption.')
        finally:
            self._shutdown_runtime(ear, world, sp_path)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _run_sleep_cycle(self, world, chem, sp_path):
        import time
        from mindai.lifecycle.sleep_consolidation import SleepPhase
        while self._sleep.is_sleeping:
            # Semantic concept tagging is an N2 phenomenon (Diekelmann 2010)
            if self._sleep.current_phase == SleepPhase.N2 and self.tick % 50 == 0:
                self._semantics.extract_concept_during_sleep(
                    self._workspace.history_buffer, self._plasticity)
                self._workspace.history_buffer.clear()

            sleep_pressure = self._clock.adenosine * 0.5 + self._clock.melatonin * 0.5
            sleep_result = self._sleep.process_sleep_tick(
                hippocampus=self._hippocampus,
                plasticity=self._plasticity,
                current_cortisol=getattr(chem, 'cortisol', 0.0),
                current_activity_np=self._activity.cpu().numpy(),
                sleep_pressure=sleep_pressure,
                pfc_monitoring_strength=self._pfc_monitoring_strength,
            )

            # Drive neuromodulators toward phase-correct biological targets (5% lerp/tick)
            targets = sleep_result.neuromod_targets
            _lerp = lambda cur, tgt: cur + (tgt - cur) * 0.05
            if self._chemistry is not None:
                self._chemistry.acetylcholine = float(
                    np.clip(_lerp(chem.acetylcholine, targets['acetylcholine']), 0.0, 1.0))
                self._chemistry.noradrenaline = float(
                    np.clip(_lerp(chem.noradrenaline, targets['noradrenaline']), 0.0, 1.0))
                self._chemistry.serotonin = float(
                    np.clip(_lerp(chem.serotonin,     targets['serotonin']),     0.0, 1.0))
                self._chemistry.anandamide = float(
                    np.clip(_lerp(chem.anandamide,    targets['anandamide']),    0.0, 1.0))

            # REM: overwrite activity with dream tensor (already on device)
            if (sleep_result.current_phase == SleepPhase.REM
                    and sleep_result.dream_tensor is not None):
                self._activity = sleep_result.dream_tensor

            # Lucid dream: dlPFC re-engages during REM (LaBerge 1985; Voss 2009)
            if sleep_result.is_lucid:
                if not getattr(self, '_was_lucid', False):
                    print(f'\n    [LUCID DREAM] Tick {self.tick} — '
                          f'dlPFC reactivated in REM. '
                          f'Monitoring={self._pfc_monitoring_strength:.3f}')
                    self._was_lucid = True
                # PFC goal biases dream activity
                pfc_goal = self._pfc.formulate_goal(
                    energy=getattr(self, '_last_h_ratio', 0.5),
                    water=getattr(self, '_last_w_ratio', 0.5),
                    base_resource=1.0,
                )
                if pfc_goal.any():
                    pfc_t = torch.tensor(
                        pfc_goal, dtype=torch.float32, device=self._device)
                    self._activity = torch.clamp(
                        self._activity + pfc_t * 0.2, 0.0, 1.0)
                # EgoModel: self-model updates during lucid REM
                act_np = self._activity.cpu().numpy()
                self._ego.evaluate_self(self.wellbeing, float(np.mean(act_np)))
            else:
                self._was_lucid = False

            # CAR: anticipatory cortisol pulse before natural wake
            if self._clock.car_cortisol_boost > 0 and self._chemistry is not None:
                self._chemistry.cortisol = float(
                    np.clip(chem.cortisol + self._clock.car_cortisol_boost * 0.01, 0.0, 1.0))

            # Phase label in mood field (visible in UI stats)
            _phase_labels = {
                SleepPhase.N1:  'Sleep N1 ( onset)',
                SleepPhase.N2:  'Sleep N2 (spindle)',
                SleepPhase.N3:  'Sleep N3 (SWS/delta)',
                SleepPhase.REM: 'REM: DREAM',
            }
            if sleep_result.current_phase in _phase_labels:
                if (sleep_result.current_phase == SleepPhase.REM
                        and sleep_result.is_lucid):
                    self.mood = 'REM: LUCID DREAM ✦'
                else:
                    self.mood = _phase_labels[sleep_result.current_phase]

            if not sleep_result.still_sleeping:
                self._clock.is_awake  = True
                self._clock.adenosine = 0.0
                self._sleep.is_sleeping = False
                # Restore awake baselines
                if self._chemistry is not None:
                    self._chemistry.acetylcholine = 0.5
                    self._chemistry.noradrenaline = 0.1
                    self._chemistry.serotonin     = 0.5
                print(f'\n>>> ORGANISM WAKING UP (Tick {self.tick}).')

            self.tick += 1
            # Control pacing to keep CPU usage reasonable while processing
            time.sleep(0.005)

    def _delete_save(self, path: str) -> None:
        """Delete brain save (called on death — world save is not deleted)."""
        d = Path(path)
        for name in ('brain.json', 'weights.npz', 'hippocampus.npz', 'entorhinal.json', 'metadata.jsonl', 'behavior.npz'):
            f = d / name
            if f.exists():
                f.unlink()
