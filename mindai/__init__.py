"""mindai — biologically-inspired artificial mind.

Learning is exclusively Hebbian/STDP. No gradient descent. No reward
functions. Behavior emerges from physiology gated by neuromodulators.

Quick start::

    from mindai import Brain
    from mindai.worlds.agent_world import AgentWorld
    from mindai.neurochemistry.neuromodulators import EndocrineSystem

    world = AgentWorld(text_corpus='corpus.txt', interactive=True)

    brain = Brain(
        num_neurons    = 500_000,
        sensory_layout = world.sensory_layout,
        motor_layout   = world.motor_layout,
        synapse_density= 0.001,
    )
    brain.attach(EndocrineSystem())
    brain.run(world, headless=True)
"""

from mindai.brain import Brain
from mindai.layout import SensoryLayout
from mindai.feels import FeelingSystem, Feel, curves
from mindai.architecture.language_cortex import LanguageCortex, WernickeModule, BrocaModule, SyntacticStack
from mindai.engine.sdr_encoder import SDREncoder
from mindai.engine.cross_modal_binder import CrossModalBinder
from mindai.architecture.theory_of_mind import TheoryOfMind, DialogueBeliefState
from mindai.architecture.prefrontal_cortex import DialogueWorkingMemory

__all__ = [
    'Brain', 'SensoryLayout', 'FeelingSystem', 'Feel', 'curves',
    'LanguageCortex', 'WernickeModule', 'BrocaModule', 'SyntacticStack',
    'SDREncoder', 'CrossModalBinder',
    'TheoryOfMind', 'DialogueBeliefState',
    'DialogueWorkingMemory',
]
__version__ = '0.5.0'
