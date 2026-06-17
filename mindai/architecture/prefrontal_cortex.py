"""Prefrontal Cortex — homeostatic goal + dialogue working memory.

Biological basis
----------------
The prefrontal cortex comprises multiple functionally distinct regions:

  vlPFC / OFC (ventrolateral, orbitofrontal — BA11/12/47):
    Encodes goal-state representations that bias downstream motor cortex
    and striatum toward deficit-reducing actions (Wallis 2007; Rushworth 2011).
    Goal representation is a distributed pattern across motor-adjacent
    neurons, not hardcoded indices.

  dlPFC (dorsolateral — BA46/9) — DialogueWorkingMemory:
    Maintains information "online" through persistent firing of layer-3
    pyramidal neurons via recurrent collaterals (Goldman-Rakic 1995).
    Crucially, dlPFC does NOT store verbatim text — it maintains compressed
    semantic representations of dialogue exchanges (D'Esposito 2007).

    Capacity: ~4 "chunks" (Cowan 2001 — revised from Miller's 7±2, which
    included sub-vocal rehearsal shortcuts not available here).

    Retrieval: cue-dependent — the current semantic state activates stored
    vectors in proportion to their cosine similarity (pattern completion
    via recurrent excitation, similar to CA3 auto-association).

    Active maintenance: each stored vector decays exponentially (τ ~ 30 s
    at 10 Hz ≈ 300 ticks) unless rehearsed by re-activation.  This matches
    the biological finding that distraction causes rapid WM loss.

    Integration with Wernicke: dlPFC back-projects to posterior temporal
    cortex (Petrides & Pandya 1988), enriching the Wernicke semantic
    computation with long-range dialogue context.

  Communicative Intent — ACC + BG:
    Separate from goal/WM: the drive to generate speech emerges from:
      * ACC detecting an information gap worth communicating
        (Botvinick 2001; Buckner 2007)
      * Mesocortical dopamine (VTA → dlPFC) indicating high-value novelty
      * DialogueBeliefState showing knowledge asymmetry vs. interlocutor
    When communicative_drive exceeds threshold, BrocaModule is triggered
    to generate an utterance (Indefrey & Levelt 2004).

References
----------
Goldman-Rakic PS (1995) Cellular basis of working memory.
  Neuron 14: 477-485.
Cowan N (2001) The magical number 4 in short-term memory.
  Behav Brain Sci 24: 87-114.
D'Esposito M (2007) From cognitive to neural models of working memory.
  Phil Trans R Soc B 362: 761-772.
Petrides M, Pandya DN (1988) Association fiber pathways to the frontal cortex.
  J Comp Neurol 271: 516-543.
Indefrey P, Levelt WJM (2004) The spatial and temporal signatures of
  word production components. Cognition 92: 101-144.
"""

from __future__ import annotations

import numpy as np


class PrefrontalCortex:
    """Prefrontal working-memory bias toward current homeostatic goal.

    Biologically: vlPFC/OFC encodes goal-state representations that bias
    downstream motor cortex and striatum toward deficit-reducing actions
    (Wallis 2007; Rushworth 2011).

    Goal representation is a distributed pattern across motor-adjacent
    neurons, not hardcoded indices — the layout slice is computed relative
    to num_nodes so the module scales across any network size.

    Two goal channels (hunger, thirst) are placed at the top 10 neurons
    of the motor region (last neurons in the array, furthest from sensory
    input, closest to output layer by convention in this layout).
    """

    def __init__(self, num_nodes: int, motor_end: int):
        self.num_nodes = num_nodes
        self.current_goal_vector = np.zeros(num_nodes, dtype=np.float32)
        self.goal_persistence = 0.0
        # Goal neuron indices: last 10 neurons of the motor region, split hunger/thirst
        # These fall within the active cortical ceiling (below 80% limit)
        self._hunger_slice = slice(max(0, motor_end - 10), max(0, motor_end - 5))
        self._thirst_slice = slice(max(0, motor_end - 5),  motor_end)

        # dlPFC dialogue working memory — attached on demand
        self.dialogue_wm: DialogueWorkingMemory | None = None

    def attach_dialogue_wm(self, semantic_dim: int) -> 'DialogueWorkingMemory':
        """Attach the dlPFC dialogue working memory module.

        Called from Brain.__init__ once LanguageCortex is instantiated.

        Parameters
        ----------
        semantic_dim : int
            Must match LanguageCortex.semantic_dim.
        """
        self.dialogue_wm = DialogueWorkingMemory(semantic_dim=semantic_dim)
        return self.dialogue_wm

    def formulate_goal(self, energy: float, water: float, base_resource: float):
        self.current_goal_vector.fill(0.0)
        if water < base_resource * 0.4 and water < energy:
            self.current_goal_vector[self._thirst_slice] = 1.0
            self.goal_persistence = 1.0
        elif energy < base_resource * 0.4:
            self.current_goal_vector[self._hunger_slice] = 1.0
            self.goal_persistence = 1.0
        else:
            self.goal_persistence *= 0.9
        return self.current_goal_vector * self.goal_persistence


# ---------------------------------------------------------------------------
# Dialogue Working Memory — dlPFC (BA46/9)
# ---------------------------------------------------------------------------

class DialogueWorkingMemory:
    """Dorsolateral PFC dialogue context buffer.

    Stores compressed semantic representations of the last N dialogue
    exchanges.  Maintenance is active (decays without rehearsal).
    Retrieval is cue-dependent via cosine similarity pattern completion.

    Parameters
    ----------
    semantic_dim : int
        Dimensionality of semantic vectors from WernickeModule.
    capacity : int
        Number of exchanges to buffer.  Cowan (2001): ~4 chunks.
        Default 6 allows for slight variation across individuals.
    decay_tau : int
        Ticks before a stored item reaches 1/e of its original strength.
        At 10 Hz, 300 ticks ≈ 30 s — matches empirical WM decay rates.
    """

    def __init__(
        self,
        semantic_dim: int,
        capacity:     int = 6,
        decay_tau:    int = 300,
    ) -> None:
        self.semantic_dim = semantic_dim
        self.capacity     = capacity
        self.decay_tau    = decay_tau

        # Buffer: list of dicts {vector, strength, role}
        # role: 'self' (agent's utterance) or 'other' (interlocutor's)
        self._buffer: list[dict] = []

        # Decay factor per tick: α = exp(-1/τ)
        self._decay = float(np.exp(-1.0 / decay_tau))

        # Recency-weighted context output (updated by retrieve())
        self.context_vector = np.zeros(semantic_dim, dtype=np.float32)

        # Communicative intent signal (ACC-gated)
        # Builds when agent has unreported novel information.
        # Decays after each successful utterance generation.
        self.communicative_drive: float = 0.0
        self._cd_decay = 0.98   # slow build-up, fast release

    # ------------------------------------------------------------------

    def store(
        self,
        semantic_vector: np.ndarray,
        role: str = 'self',   # 'self' or 'other'
    ) -> None:
        """Store a new exchange in working memory.

        Replaces the oldest item when at capacity.  Biologically:
        encoding into dlPFC WM requires attentional gating (NA/ACh).

        Parameters
        ----------
        semantic_vector : np.ndarray [semantic_dim]
        role : str  — who produced this utterance
        """
        v = _fit(semantic_vector, self.semantic_dim)
        entry = {'vector': v.copy(), 'strength': 1.0, 'role': role}

        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)   # FIFO eviction of oldest item
        self._buffer.append(entry)

    def decay(self) -> None:
        """Apply one tick of exponential decay to all stored items.

        Should be called once per brain tick.
        Biologically: passive decay via leaky integrate-and-fire dynamics;
        prevented by active maintenance (recurrent collateral firing).
        Items that decay to < 0.05 are pruned (forgotten).
        """
        survivors = []
        for entry in self._buffer:
            entry['strength'] *= self._decay
            if entry['strength'] > 0.05:
                survivors.append(entry)
        self._buffer = survivors

    def rehearse(self, semantic_vector: np.ndarray) -> None:
        """Refresh the closest matching item in WM (active maintenance).

        Biologically: dlPFC recurrent collaterals re-excite the stored
        pattern when it is retrieved, resetting its decay clock.
        Called when the agent's current semantic state is similar to a
        buffered item — the item is "refreshed" (strength → 1.0).

        Parameters
        ----------
        semantic_vector : np.ndarray [semantic_dim]
            Current query (e.g. from Wernicke semantic_vector).
        """
        if not self._buffer:
            return
        q = _fit(semantic_vector, self.semantic_dim)
        best_idx, best_sim = 0, -1.0
        for i, entry in enumerate(self._buffer):
            sim = float(np.dot(q, entry['vector']) / (
                np.linalg.norm(q) * np.linalg.norm(entry['vector']) + 1e-8))
            if sim > best_sim:
                best_sim, best_idx = sim, i
        if best_sim > 0.5:   # threshold: only rehearse if genuinely similar
            self._buffer[best_idx]['strength'] = min(1.0,
                self._buffer[best_idx]['strength'] + 0.2)

    def retrieve(self, query_vector: np.ndarray) -> np.ndarray:
        """Retrieve a context-weighted recall vector.

        Cue-dependent retrieval: items are weighted by cosine similarity
        to the query AND by their current maintenance strength.
        Biologically: CA3-like pattern completion through recurrent
        excitation in dlPFC layer 3 pyramidal circuits.

        Parameters
        ----------
        query_vector : np.ndarray [semantic_dim]
            Current semantic state (Wernicke output).

        Returns
        -------
        context_vector : np.ndarray [semantic_dim]
            Weighted recall of relevant past exchanges.
            Zero vector if buffer is empty.
        """
        if not self._buffer:
            self.context_vector = np.zeros(self.semantic_dim, dtype=np.float32)
            return self.context_vector

        q = _fit(query_vector, self.semantic_dim)
        weights = np.zeros(len(self._buffer), dtype=np.float32)

        for i, entry in enumerate(self._buffer):
            sim = float(np.dot(q, entry['vector']) / (
                np.linalg.norm(q) * np.linalg.norm(entry['vector']) + 1e-8))
            weights[i] = max(0.0, sim) * entry['strength']

        total_w = weights.sum()
        if total_w < 1e-8:
            self.context_vector = np.zeros(self.semantic_dim, dtype=np.float32)
            return self.context_vector

        weights /= total_w
        recall = np.zeros(self.semantic_dim, dtype=np.float32)
        for i, entry in enumerate(self._buffer):
            recall += weights[i] * entry['vector']

        self.context_vector = recall
        return recall

    def update_communicative_drive(
        self,
        surprise:         float,   # Brain.surprise — novelty signal
        knowledge_gap:    float,   # DialogueBeliefState.knowledge_gap
        dopamine_meso:    float,   # mesocortical DA — cognitive flexibility
        just_spoke:       bool,    # agent produced an utterance this tick
    ) -> float:
        """Update communicative intent signal (ACC-gated).

        Biological basis (Botvinick 2001; Indefrey & Levelt 2004):
          * ACC detects the gap between what is known and what is expressed.
          * High mesocortical DA → high willingness to communicate new info.
          * Knowledge gap (from DialogueBeliefState): if interlocutor doesn't
            know something the agent knows → drive increases.
          * After speaking, drive resets (information has been transmitted).

        Parameters
        ----------
        surprise : float [0, ∞)
            Current neural surprise — high surprise = novel information.
        knowledge_gap : float [0, 1]
            From DialogueBeliefState: how much does interlocutor not know.
        dopamine_meso : float [0, 1]
            Mesocortical dopamine — scales willingness to initiate speech.
        just_spoke : bool
            True if utterance was generated this tick → drives resets.

        Returns
        -------
        communicative_drive : float [0, 1]
        """
        if just_spoke:
            self.communicative_drive *= 0.1   # transmitted → rapid decay
        else:
            # Accumulate: novelty × knowledge_gap × DA-gated
            increment = (
                float(np.clip(surprise / 10.0, 0.0, 1.0)) *
                knowledge_gap *
                dopamine_meso *
                0.05
            )
            self.communicative_drive = float(np.clip(
                self.communicative_drive * self._cd_decay + increment,
                0.0, 1.0))

        return self.communicative_drive

    # ------------------------------------------------------------------

    @property
    def n_items(self) -> int:
        """Number of items currently in working memory."""
        return len(self._buffer)

    @property
    def mean_strength(self) -> float:
        """Mean maintenance strength of buffered items."""
        if not self._buffer:
            return 0.0
        return float(np.mean([e['strength'] for e in self._buffer]))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate 1-D float32 array to length n."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))