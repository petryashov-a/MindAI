<div align="center">

# MindAI

### A spiking neural network that learns language through Hebbian plasticity

*No gradient descent. No loss functions. No backpropagation.*
*Synapses, neuromodulators, and time.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20(grad%20disabled)-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20research-orange?style=flat-square)]()

</div>

---

## What this project is

MindAI is a research project exploring whether an LLM-like language agent can
emerge from biologically accurate neural mechanisms — without any form of
gradient-based learning.

`torch.set_grad_enabled(False)` is set at startup. There is no optimizer, no
backpropagation, no loss function, and no global error signal anywhere in the
codebase. Plasticity is local and Hebbian: synapses change because the neurons
they connect fired in close temporal proximity (Hebb 1949; Bi & Poo 1998).
Neuromodulators *gate* how much plasticity occurs, never *which direction*
to update — there is no teacher.

The system is built around a recurrent sparse spiking network of n count neurons running on a single GPU. On top of that substrate, ~30 modules
implement specific anatomical structures of the human brain (PFC, amygdala,
hippocampal subfields DG/CA3/CA1, cerebellum, basal ganglia, periaqueductal gray,
locus coeruleus, etc.). Each module is documented against the peer-reviewed paper
it implements.

The entry point — `main_agent.py` — uses this substrate as a **multimodal
language analog**: text, images with captions, video with audio, and Q&A pairs
are presented to the agent and learned through Hebbian binding between sensory
channels. Memory lives in synaptic weights, not in a context window;
conversation and training are the same loop.

Crucially, the current token channel is **character-level**, not BPE/subword.
This is more biologically plausible for this project: cortex does not receive
pre-segmented "perfect subword units" from a tokenizer trained offline. It must
bind smaller perceptual units over time into stable lexical assemblies.
Character streams let words with small spelling variations co-activate strongly
overlapping neuron populations, which is closer to how distributed cortical
representations generalize. For example, `home` and `hhome` share the same prefix
characters and therefore drive much of the same pathway, so the learned concept
assembly can still activate under minor misspelling instead of fragmenting into
an unrelated token. That improves robustness to noisy spelling while staying
closer to a biologically grounded incremental binding story.

---

## Position relative to other work

This project sits in computational neuroscience and neuromorphic computing, not in
the transformer / statistical-ML tradition.

|                              | Transformer / LLM                                       | MindAI                                                        |
| ---------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
| Learning rule                | Backpropagation through a differentiable computation graph | Spike-Timing-Dependent Plasticity, local at each synapse      |
| Objective                    | Minimize a loss on a dataset                            | None. Behavior emerges from physiology under sensory pressure |
| Context window               | Fixed-length attention window                           | None. Context lives in synaptic weights, integrated over the agent's lifetime |
| Computation graph            | Layered (embed → attention → MLP × N)                   | Recurrent sparse matrix over anatomical regions               |
| Training vs. inference       | Separate phases                                         | Same loop. Synapses change on every tick, including during chat |
| Time                         | Discrete positional encoding                            | Continuous: axonal delays, refractory periods, theta/gamma windows |

Within computational neuroscience, the closest relatives are Nengo / Spaun
(Eliasmith et al.), Blue Brain (Markram), Numenta HTM (Hawkins), and the spiking
networks targeted by Intel Loihi and SpiNNaker. The contribution of MindAI is not
the underlying paradigm — SNN + STDP is decades old — but the **scope of integration**:
most published SNN models examine a single mechanism in isolation. MindAI integrates
~30 such mechanisms in a single running process, sharing one synaptic substrate, on
a clock.

---

## Architectural overview

```
┌────────────────────────────────────────────────────────────────┐
│  AgentWorld — multimodal sensory stream + curriculum            │
│  text · images+captions · video+audio · Q&A · interactive chat │
└────────────┬───────────────────────────────────────────────────┘
             │
   ┌─────────▼──────────────┐
   │  FovealRetina          │  non-uniform foveal sampling (Curcio 1990)
   │  Cochlea               │  ERB-spaced basilar membrane (Glasberg & Moore 1990)
   │  Token channel         │  Character-level tokenizer (Russian + English)
   └─────────┬──────────────┘
             │ packed sensory vector
   ┌─────────▼─────────────────────────────────────────────────┐
   │  Sparse recurrent connectome   400k–1.5M neurons, GPU    │
   │  80% glutamatergic · 20% GABAergic (Dale's principle)    │
   │  Synapse density 2×10⁻⁴ · STDP weights · axonal delays   │
   └─────────┬─────────────────────────────────────────────────┘
             │ recurrent dynamics + delayed PSPs
   ┌─────────▼──────────────┐
   │  PredictiveMicro-      │  L5/6 prediction · L2/3 error
   │  circuits              │  (Rao & Ballard 1999)
   └─────────┬──────────────┘
             │ surprise scalar
   ┌─────────▼──────────────┐
   │  HusserlianTime        │  retention · primal impression · protention
   │                        │  (theta-gamma temporal binding, Lisman & Jensen 2013)
   └─────────┬──────────────┘
             │
   ┌─────────▼──────────────┐
   │  Thalamus              │  attention gate (NA-driven), (Crick 1984)
   └─────────┬──────────────┘
             │ salient signal
   ┌─────────▼──────────────────────────────┐
   │  Phase-Coupled Workspace (Kuramoto)    │  ignition when R > 0.7
   │                                        │  (Baars 1988; Dehaene 2001)
   └────┬───────────────────────────┬───────┘
        │ broadcast                 │ snapshot
        │                  ┌────────▼────────────────┐
        │                  │ Hippocampus DG/CA3/CA1  │  pattern separation +
        │                  │                         │  completion (Marr 1971;
        │                  │                         │  Treves & Rolls 1994)
        │                  └─────────────────────────┘
   ┌────▼───────────────────────────────────────────────────────┐
   │  EndocrineSystem — 15 neuromodulators                      │
   │  3 dopamine pathways (mesolimbic, mesocortical,            │
   │  nigrostriatal) · 5-HT · NA · cortisol · oxytocin ·        │
   │  endorphins · adrenaline · ACh · anandamide ·              │
   │  substance P · ghrelin · leptin · vasopressin ·            │
   │  prolactin · insulin                                       │
   └────┬───────────────────────────────────────────────────────┘
        │ chemical state
   ┌────▼──────────────────────────────────────────────────────┐
   │  Action selection                                          │
   │  BasalGanglia (D1 direct · D2 indirect · STN hyperdirect) │
   │    Δw = η · pre · post · (DA − baseline)  (Reynolds 2002) │
   │  ACC conflict monitor → STN brake (Botvinick 2001)        │
   │  Cerebellum forward model + climbing-fibre LTD (Ito 1984) │
   │  FreeWillEngine — Libet delay + somatic veto (Libet 1983) │
   └────┬──────────────────────────────────────────────────────┘
        │ motor pattern
   ┌────▼───────────────────────┐
   │  SuperiorColliculus        │  retinotopic saccade map →
   │                            │  emergent gaze (no scripts)
   └────┬───────────────────────┘
        │
   ┌────▼──────────────────────────────────────────────────────┐
   │  Plasticity — STDP gated by DA / cortisol / ACh           │
   │  Astrocytes (Volterra 2005) · structural plasticity ·     │
   │  neurogenesis on surprise · Turrigiano homeostatic scaling│
   └───────────────────────────────────────────────────────────┘
```

Sleep is a separate pathway: when adenosine and melatonin cross threshold
(Borbély 1982), the main loop is bypassed and the `SleepCycle` replays
hippocampal episodes through structural plasticity at 5× rate. High cortisol
during sleep stochastically corrupts replayed memories — a model of
trauma-influenced consolidation (Diekelmann & Born 2010).

---

## Selected mechanisms

### Synaptic plasticity — local Hebbian/STDP
Weights change exclusively through spike-timing-dependent plasticity
(Bi & Poo 1998). No global error signal exists. Neuromodulators multiply the
magnitude of change; they cannot reverse its direction.

```
ΔW_LTP  ∝  pre_trace(t) · post_active(t) · m(chemistry)
ΔW_LTD  ∝  pre_active(t) · post_trace(t) · m(chemistry)

m(chemistry) = (DA · 1.5 + 5HT · 0.5) · (1 − cortisol) · (1 + endorphins)
```

Turrigiano homeostatic scaling normalizes total incoming weight when neurons
become overloaded (Turrigiano 2008). Additionally, Astrocytes track running average neural
activity via a slow EMA: hyperactive neurons (>8% average firing) scale down incoming
synapses by 0.999/tick, while underactive neurons (<1% average firing) scale up by 1.001/tick.

### Three-factor corticostriatal learning
Action selection follows the canonical three-factor rule:

```
Δw = η · pre · post · (DA − baseline)
```

D1 (direct) and D2 (indirect) pathways use mirrored update rules so that
above- and below-baseline dopamine drive approach and avoidance respectively
(Reynolds & Wickens 2002). The subthalamic hyperdirect pathway provides a fast
brake gated by anterior cingulate conflict monitoring (Botvinick et al. 2001).

### Predictive hierarchy
Two anatomically distinct neuron populations:

- Prediction neurons (L5/6 analogue): `Ŷ = W_td · internal_state`
- Error neurons (L2/3 analogue): `ε = sensory − Ŷ` (signed, bidirectional)

Both populations are persistent across ticks, so other modules can read the
current prediction state or surprise signal independently. Surprise drives
acetylcholine release, which sharpens STDP precision and triggers neurogenesis
(Rao & Ballard 1999; Kilgard & Merzenich 1998).

### Global Workspace — Kuramoto ignition
Phase coupling is computed across active neurons via the Kuramoto order
parameter R. When R exceeds 0.7, a nonlinear ignition event amplifies activity
×3 and broadcasts it globally, with a 30-tick refractory period — the
all-or-nothing conscious access reported by Dehaene & Changeux (2011).

### Emergent gaze — Superior Colliculus
There is no scripted saccade controller. The `SuperiorColliculus` builds a
retinotopic priority map each tick:

```
spatial_sal = motion · (0.3 + 0.7·ACh) + luma · NA + surprise · NA / 10
              × (1 + 3 · threat)
priority    = spatial_sal · (1 − IOR) + goal · 0.08
```

`luma` is read from the post-recurrent activity of visual neurons, which
includes top-down feedback. Hearing the token "tree" → activates the concept
neurons → STDP-learned back-connections fire visual neurons that previously
co-occurred with trees → priority elevates at tree locations → the eye saccades
there. This is voluntary, language-directed gaze emerging from the substrate,
not a Python control flow.

### Continuous time
Axonal delays are modeled per-synapse via a GPU ring buffer (Swadlow 1985).
Short-term synaptic plasticity (Tsodyks & Markram 1997), refractory periods
(Hodgkin & Huxley 1952), spike-frequency adaptation, and k-WTA lateral
inhibition (Buzsáki 2004) give the simulation continuous-time dynamics rather
than discrete layer-by-layer updates.

### 15-modulator endocrine system
| Modulator       | Source                | Primary role                                   |
|-----------------|-----------------------|------------------------------------------------|
| Dopamine (3 paths) | VTA, SN            | Reward prediction · cognitive gating · movement |
| Serotonin       | Raphe / gut           | Mood baseline · post-meal satiety              |
| Noradrenaline   | Locus coeruleus       | Thalamic gain · arousal · SC orienting         |
| Cortisol        | HPA axis              | Stress · synaptic damage at sustained high levels |
| Oxytocin        | Hypothalamus          | Social trust · mirror neuron up-regulation     |
| Endorphins      | Pituitary / PAG       | Analgesia · post-reward plasticity boost       |
| Adrenaline      | Adrenal medulla       | Fight-or-flight                                |
| Acetylcholine   | Basal forebrain       | STDP precision gate · novelty                  |
| Anandamide      | Endocannabinoid       | CB1 → CA1 LTP suppression (forgetting)         |
| Substance P     | Spinal DRG / PAG      | Two-phase pain wind-up                         |
| Ghrelin         | Stomach X/A cells     | Pre-meal VTA dopamine drive                    |
| Leptin          | Adipocytes            | Post-meal satiety                              |
| Vasopressin     | Hypothalamus / SON    | Territorial vigilance                          |
| Prolactin       | Anterior pituitary    | Post-stress affiliative drive                  |
| Insulin         | Pancreatic β-cells    | Post-meal → tryptophan → serotonin             |

---

## Cross-modal grounding

Concepts are learned by simultaneous activation across sensory channels — STDP
binds whichever neurons happen to be coactive. A video that says "this is a
tree" while showing a tree produces:

```
Cochlea[tree_phonemes] + FovealRetina[tree_pixels]
       → STDP binds [tree_pixels] ↔ [tree_phonemes]

FovealRetina[tree_pixels] + Token[derevo]   (caption "это дерево")
       → STDP binds [tree_pixels] ↔ [token_derevo]

FovealRetina[pixels_of_the_word_"дерево"] + Token[derevo]
       → STDP binds the visual word-form to the same token
```

After training, all four representations (sight, sound, spoken token, written
token) converge on the same neuron cluster. Asking "что рядом с деревом?"
activates the cluster, which through learned top-down connections elevates
luma in tree-shaped regions of the retinal priority map — and the eye looks
there.

---

## Module map

```
mindai/
├── brain.py                              Main tick loop, module orchestration
├── layout.py                             SensoryLayout — named channels → neuron indices
│
├── engine/
│   ├── plasticity_core.py                STDP, STP, refractory, adaptation, lateral inhibition
│   ├── temporal_windows.py               Theta-gamma temporal binding (Lisman & Jensen 2013)
│   ├── axonal_delays.py                  Per-synapse axonal delay queue (Swadlow 1985)
│   ├── sdr_encoder.py                    Sparse Distributed Representation (SDR) token encoding
│   └── cross_modal_binder.py             Hebbian cross-modal binding (Damasio 1989 convergence zones)
│
├── architecture/
│   ├── predictive_hierarchy.py           Rao & Ballard 1999 predictive coding
│   ├── thalamocortical_core.py           Crick 1984
│   ├── prefrontal_cortex.py              Wallis 2007 (has DialogueWorkingMemory)
│   ├── language_cortex.py                Broca production + Wernicke comprehension (dorsal stream)
│   ├── cortical_areas.py / cortical_layers.py  Zeki 1978; Elston 2003
│   ├── hippocampus_buffer.py             Scoville & Milner 1957
│   ├── hippocampus_subfields.py          DG/CA3/CA1 pattern separation (Marr 1971)
│   ├── entorhinal.py                     Grid cells (Hafting 2005)
│   ├── amygdala.py                       Dual-path fear (LeDoux 1996)
│   ├── habenula.py                       Anti-reward (Matsumoto & Hikosaka 2007)
│   ├── pag.py                            Fight / flight / freeze (Bandler & Shipley 1994)
│   ├── insula.py                         Interoception (Craig 2002)
│   ├── anterior_cingulate.py             Conflict monitoring (Botvinick 2001)
│   ├── cerebellum.py                     Forward model + climbing-fibre LTD (Ito 1984)
│   ├── superior_colliculus.py            Retinotopic saccade map (Wurtz & Goldberg 1989)
│   ├── biological_motion_detector.py     MT+/V5 (Grossman & Blake 2002)
│   ├── default_mode_network.py           Autobiographical replay (Raichle 2001)
│   ├── visuospatial_sketchpad.py         Spatial working memory (Baddeley 1986)
│   ├── theory_of_mind.py                 Other-agent modeling (Saxe 2003)
│   ├── semantic_memory.py                Concept extraction during sleep (Diekelmann 2010)
│   ├── astrocytes.py                     Glial weight stability (Volterra 2005)
│   └── olfactory_bulb.py                 Non-thalamic sensory route (Buck & Axel 1991)
│
├── consciousness/
│   ├── global_workspace.py               Kuramoto ignition (Baars 1988; Dehaene 2001)
│   ├── self_model_ego.py                 Sense of agency (Wegner 1999)
│   ├── volition_and_agency.py            BasalGanglia + FreeWillEngine
│   └── neural_complexity.py              LZ76 + spectral radius (Casali 2013)
│
├── neurochemistry/
│   └── neuromodulators.py                15-modulator endocrine system
│
├── lifecycle/
│   ├── sleep_consolidation.py            NREM/REM, replay, lucid dreaming (Diekelmann 2010)
│   └── circadian_rhythm.py               Adenosine + melatonin (Borbély 1982)
│
├── environment/
│   ├── hearing_system.py                 ERB cochlea (Glasberg & Moore 1990)
│   └── retina.py                         FovealRetina — non-uniform foveal vision (Curcio 1990)
│
├── feels/                                FeelingSystem — Stevens 1957, Weber-Fechner
├── speech/                               Vocal apparatus + ear (faster-whisper)
└── worlds/
    ├── agent_world.py                    Multimodal training + chat (primary)
    └── tokenizers/                       Character tokenizer API
```

---

## Installation and running

```bash
git clone https://github.com/mellson19/mindai.git
cd mindai
python -m venv .venv
.venv\Scripts\activate                     # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e .

# Run the multimodal agent (LLM analog)
python main_agent.py                       # stdin chat + training
python main_agent.py --data /other/path    # custom data directory
python main_agent.py --c4                  # stream from allenai/c4 (pip install datasets)
python main_agent.py --rehab-ticks N       # optional rehabilitation-mode run length
```

**Requirements:** Python 3.10+, PyTorch (CUDA strongly recommended for ≥400k neurons),
NumPy, SciPy. Optional: `datasets` for `--c4`, `av` or `moviepy` for video,
`faster-whisper` for speech I/O.

---

## Data layout for multimodal training

```
data/
├── corpus.txt                paragraph-separated text
├── qa.txt                    question / answer pairs (alternating lines)
├── images/                   images with optional .txt captions
│   ├── chair.jpg
│   └── chair.txt             "I see a chair. This is a chair."
├── video/                    videos with optional .txt captions
│   ├── lecture.mp4
│   └── lecture.txt
└── audio/                    standalone audio files
```

## Curriculum: strict phase machine, no modality mixing

Training in `mindai/worlds/agent_world.py` is **strictly phase-based**. Exactly
**one curriculum source is active at a time**; the world does not mix text,
images, video, and Q&A into one simultaneous stream.

Architecturally, the curriculum progresses through the canonical order
`text → images → video → qa`, with each phase activating only the sensory
channels relevant to that source:

| Phase  | Active source | Biological purpose |
|--------|---------------|--------------------|
| `text`   | `corpus.txt` or streamed text (`--c4`) | Build sequence statistics and predictive token dynamics |
| `images` | `images/` + captions | Hebbian visual-text grounding via ostensive definition |
| `video`  | `video/` + captions + extracted audio | Bind motion, sound, and text in one multisensory episode |
| `qa`     | `qa.txt` | Build dialogue structure on top of grounded concepts |

This design is deliberate. STDP binds whatever fires together in time, so
parallel "everything at once" training would create **spurious bindings**:
random corpus tokens could become associated with background audio, wall colors,
or unrelated video frames simply because they co-occurred on the same tick.
The phase machine prevents that by isolating modalities.

Key implementation rules in the current world:

- **No spurious binding:** each phase silences channels that do not belong to it.
  During text/Q&A phases, paired vision is inactive. During non-video phases,
  the audio channel is zeroed. During paired media episodes, corpus tokens do
  not leak into the token stream.
- **Exactly one active source:** the curriculum order is `text → images → video → qa`.
  A new source is chosen only when the current token queue is empty, so the
  world does not interrupt a sentence mid-stream.
- **Automatic skipping:** if `images/`, `video/`, or `qa.txt` are missing, the
  world automatically skips that phase instead of wasting ticks.
- **Configurable scheduling:** phase boundaries are user-configurable at world
  construction time via `phase_ticks={...}`; the phase machine itself is the
  architectural invariant, not any particular numeric schedule.

This phase isolation is scientifically important for MindAI. In a local
Hebbian/STDP system, learning quality depends not only on what patterns are
present, but on which patterns are allowed to be co-active. Preventing false
co-activation is therefore part of the learning rule, not just a data-loading
implementation detail.

---

## Save format

```
savegame_brain/
├── brain.json                Main metadata (tick, active_limit, mood, rehab_ticks_left)
├── metadata.jsonl            Waking phase statistics log
├── weights.npz               Connectome weights + astrocytes, language, pred values
├── behavior.npz              Basal ganglia and Pavlovian behavior weights
├── hippocampus.npz           Episodic patterns stored in CA3/CA1 subfields
└── entorhinal.json           Learned semantic positions of conceptual spaces
```

---

## What this is not

- **Not a deep learning model.** No gradient descent anywhere in the codebase.
  Plasticity is local; the global error signal that defines deep learning does
  not exist here.
- **Not a reinforcement learning agent.** Dopamine is a physiological state
  variable driven by predicted homeostatic change, not a reward channel set by
  the developer. Reward shaping is forbidden by project rules.
- **Not a transformer.** No attention layers, no positional encoding, no
  layered embed/MLP stack. The computational graph is a recurrent sparse matrix
  over anatomically defined regions.
- **Not biologically complete.** ~30 modules cover only a tiny fraction of what
  the brain does. Glial dynamics are simplified, glutamate / GABA are the only
  two transmitter classes treated explicitly at the synapse level, and axonal
  delay realism is capped at 20 ticks.
- **Not benchmarked against LLMs.** This is research into whether the right
  combination of biological mechanisms produces emergent cognition in a single
  system. It is not, and may never be, competitive with statistical
  language models on standard NLP tasks. That is by design — different
  question, different tools.

---

## Known limitations (tracked, not hidden)

- STDP `pre_trace` and `post_trace` are both written to 1.0 within the same
  tick, partially erasing temporal ordering — to be replaced with a continuous
  decaying eligibility trace.

---

## Key references

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.
- Bi, G., & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24).
- Borbély, A. A. (1982). A two-process model of sleep regulation. *Human Neurobiology*, 1(3).
- Botvinick, M. M., et al. (2001). Conflict monitoring and cognitive control. *Psychological Review*, 108(3).
- Casali, A. G., et al. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. *Science Translational Medicine*, 5(198).
- Craig, A. D. (2002). How do you feel? Interoception. *Nature Reviews Neuroscience*, 3(8).
- Curcio, C. A., et al. (1990). Human photoreceptor topography. *Journal of Comparative Neurology*, 292(4).
- Damasio, A. (1994). *Descartes' Error.* Putnam.
- Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2).
- Diekelmann, S., & Born, J. (2010). The memory function of sleep. *Nature Reviews Neuroscience*, 11(2).
- Glasberg, B. R., & Moore, B. C. J. (1990). Derivation of auditory filter shapes from notched-noise data. *Hearing Research*, 47.
- Hafting, T., et al. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*, 436.
- Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current. *Journal of Physiology*, 117(4).
- Ito, M. (1984). *The Cerebellum and Neural Control.* Raven Press.
- LeDoux, J. (1996). *The Emotional Brain.* Simon & Schuster.
- Libet, B., et al. (1983). Time of conscious intention to act in relation to onset of cerebral activity. *Brain*, 106(3).
- Lisman, J. E., & Jensen, O. (2013). The theta-gamma neural code. *Neuron*, 77(6).
- Marr, D. (1971). Simple memory: a theory for archicortex. *Phil. Trans. R. Soc. B*, 262(841).
- Matsumoto, M., & Hikosaka, O. (2007). Lateral habenula as a source of negative reward signals. *Nature*, 447.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1).
- Reynolds, J. N. J., & Wickens, J. R. (2002). Dopamine-dependent plasticity of corticostriatal synapses. *Neural Networks*, 15(4–6).
- Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical pyramidal neurons. *PNAS*, 94(2).
- Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. *Cell*, 135(3).
- Wurtz, R. H., & Goldberg, M. E. (1989). *The Neurobiology of Saccadic Eye Movements.* Elsevier.

---

<div align="center">

*The question is not whether machines can think.*
*The question is whether thinking can emerge from the right kind of physics.*

</div>
