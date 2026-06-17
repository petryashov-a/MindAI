"""Theory of Mind (ToM) — other agents' mental state modeling.

Biological basis (Premack & Woodruff 1978; Baron-Cohen 1985; Frith 1992):
  ToM (also called mentalising) is the ability to attribute mental states
  (beliefs, desires, intentions) to other agents and use these to predict
  and explain their behaviour.

  Neural substrate (Saxe & Kanwisher 2003; Frith & Frith 2006):
    TPJ (temporoparietal junction) — right TPJ is consistently activated
      when attributing beliefs to others.  Damage → inability to distinguish
      intentional from accidental actions.
    mPFC (medial prefrontal, area 10) — represents mental states of others;
      also active in self-reflection (shared circuitry: Frith 2007).
    STS (posterior, right hemisphere) — biological motion → intentional agent.
    Temporal poles — semantic knowledge of person-specific behaviour.

  First-order ToM: "I believe that X believes that..."
  Second-order ToM: "I believe that X believes that Y believes..."
  Children develop first-order at ~3–4 years, second-order at ~5–6 years.

  Link to mirror neurons (Gallese & Goldman 1998 — simulation theory):
    Mirror neurons provide the motor simulation of another's action.
    ToM adds the *inference* about the goal/intention behind that action.
    Shared representation hypothesis: understanding actions via simulation,
    then attributing goals = low-level ToM.

Physical ToM (existing TheoryOfMind class):
  The agent observes the human's action history (from world signals).
  A simple agent model: maintained belief about human's current goal state,
  updated by observed human actions.

  No scripted "theory" — the agent builds up statistical associations
  between observed human behaviours via Hebbian association in this module.
  The belief state is a soft probability distribution over possible
  human "modes" (approaching, retreating, idle, attacking).

  Association weights update Hebbian-style: when the agent observes
  a behaviour AND has a strong prediction, co-activation strengthens
  the relevant association.

Dialogue ToM (DialogueBeliefState — new):
  An extension for *linguistic* interactions.  Goes beyond physical
  modes to model:
    1. What topic the interlocutor is currently discussing
       (interlocutor_semantic_state — a vector in semantic space)
    2. What the interlocutor believes the agent knows
       (belief_about_other_knowledge — estimated knowledge gaps)
    3. Simulation of the interlocutor's affect using the agent's own
       FeelingSystem as a template (Gallese & Goldman 1998 — simulation theory)

  Biological basis:
    mPFC area 10 (Frith 2007): holds a model of the other's
      beliefs/intentions updated by each utterance.
    Right TPJ (Saxe & Kanwisher 2003): updated when the interlocutor's
      utterance is semantically surprising (misalignment between
      expected and observed semantic vector).
    Angular gyrus: semantic integration across modalities and turns.

  The DialogueBeliefState does NOT require explicit symbolic logic —
  it uses the same Hebbian prediction-error mechanism as the rest of the
  architecture, applied to semantic vectors from LanguageCortex.Wernicke.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# Human behavioural modes inferred from world signals
_MODES = ['approaching', 'retreating', 'idle', 'interacting', 'threat']


class TheoryOfMind:

    def __init__(self, seed: int = 42):
        n = len(_MODES)
        # Belief distribution over human modes — uniform prior
        self.belief: np.ndarray = np.ones(n, dtype=np.float32) / n

        # Association weights: observable features → mode probabilities
        # Features: [distance_change, proximity, interaction_flag, threat_signal]
        rng = np.random.default_rng(seed)
        self._assoc_w = rng.uniform(0.1, 0.3, (n, 4)).astype(np.float32)

        # Smoothed prediction confidence
        self.confidence: float = 0.0

        self._prev_distance: float = float('inf')

        # Dialogue belief state (optional — populated when LanguageCortex is available)
        self.dialogue: DialogueBeliefState | None = None

    def attach_dialogue_belief(self, semantic_dim: int) -> 'DialogueBeliefState':
        """Attach a DialogueBeliefState for linguistic ToM.

        Called from Brain.__init__ once LanguageCortex is instantiated.

        Parameters
        ----------
        semantic_dim : int
            Must match LanguageCortex.semantic_dim.

        Returns
        -------
        DialogueBeliefState (also stored as self.dialogue)
        """
        self.dialogue = DialogueBeliefState(semantic_dim=semantic_dim)
        return self.dialogue

    def update(
        self,
        distance:          float,     # distance to human (world units)
        human_interacted:  bool,      # human pressed interaction key
        threat_signal:     float,     # amygdala threat level
    ) -> dict:
        """Update belief about human mental state.

        Returns:
          belief         : np.ndarray [5] — distribution over human modes
          most_likely    : str — highest-probability mode label
          confidence     : float [0,1] — entropy-based confidence
        """
        # Build feature vector from observable signals
        dist_change  = float(np.clip(
            (self._prev_distance - distance) / 10.0, -1.0, 1.0))
        proximity    = float(np.clip(1.0 - distance / 50.0, 0.0, 1.0))
        interact_f   = 1.0 if human_interacted else 0.0
        features     = np.array([dist_change, proximity, interact_f, threat_signal],
                                 dtype=np.float32)

        # Likelihood: dot product of features with association weights
        likelihood = np.dot(self._assoc_w, features)
        likelihood = np.exp(likelihood - likelihood.max())   # softmax
        likelihood /= likelihood.sum() + 1e-9

        # Bayesian update: posterior ∝ prior × likelihood
        posterior = self.belief * likelihood
        posterior /= posterior.sum() + 1e-9
        self.belief = 0.85 * self.belief + 0.15 * posterior

        # Hebbian weight update: co-activate features with current belief
        if self.belief.max() > 0.5:
            dominant = np.argmax(self.belief)
            self._assoc_w[dominant] = np.clip(
                self._assoc_w[dominant] + features * 0.005, 0.0, 1.0)

        # Confidence: 1 − normalised entropy
        p = self.belief / (self.belief.sum() + 1e-9)
        p = np.clip(p, 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(_MODES))
        self.confidence = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

        self._prev_distance = distance

        return {
            'belief':       self.belief,
            'most_likely':  _MODES[int(np.argmax(self.belief))],
            'confidence':   self.confidence,
        }


# ---------------------------------------------------------------------------
# DialogueBeliefState — linguistic Theory of Mind
# ---------------------------------------------------------------------------

class DialogueBeliefState:
    """First-order linguistic Theory of Mind for dialogic agents.

    Models three quantities about the interlocutor:
      1. ``interlocutor_topic`` — running semantic vector of what they are
         talking about (maintained via exponential smoothing of their
         utterance embeddings).
      2. ``knowledge_gap`` — estimated difference between what the agent
         knows (its own semantic_vector) and what the interlocutor appears
         to know (their observed semantic history).  High gap → agent should
         explain; low gap → topic can be assumed shared.
      3. ``simulated_affect`` — estimate of the interlocutor's emotional
         state, computed by projecting their utterance pattern through the
         agent's own feeling-sensitivity matrix (Gallese & Goldman 1998).

    All updates are purely Hebbian — no backpropagation.

    Parameters
    ----------
    semantic_dim : int
        Must match LanguageCortex.semantic_dim (WernickeModule output dim).
    history_len : int
        Number of interlocutor utterance embeddings to buffer.
    """

    def __init__(self, semantic_dim: int, history_len: int = 16) -> None:
        self.semantic_dim  = semantic_dim
        self.history_len   = history_len

        # Running estimate of interlocutor's current topic
        self.interlocutor_topic = np.zeros(semantic_dim, dtype=np.float32)

        # Belief about what the interlocutor knows (soft estimate)
        # Initialised to a broad uniform distribution (maximal uncertainty)
        self.belief_about_other_knowledge = np.full(
            semantic_dim, 0.5, dtype=np.float32)

        # Simulated affect: estimated valence/arousal of interlocutor
        # Format: [valence (−1..1), arousal (0..1)]
        self.simulated_affect = np.array([0.0, 0.5], dtype=np.float32)

        # Prediction error: how surprising was their last utterance?
        self.mentalising_error: float = 0.0

        # Utterance history buffer
        self._history: deque[np.ndarray] = deque(maxlen=history_len)

        # Hebbian weight: agent's own semantic → interlocutor's expected semantic
        # (learned by observing what they say in response to what the agent says)
        rng = np.random.default_rng(77)
        self._W_mentalise = (
            rng.standard_normal((semantic_dim, semantic_dim)).astype(np.float32) * 0.01)

        # Feeling sensitivity: maps interlocutor semantic → [valence, arousal]
        # Shared with the agent's own interoceptive mapping (simulation theory)
        self._W_affect = rng.standard_normal((2, semantic_dim)).astype(np.float32) * 0.01

    # ------------------------------------------------------------------

    def update_from_utterance(
        self,
        observed_semantic:  np.ndarray,    # WernickeModule output for interlocutor's turn
        agent_semantic:     np.ndarray,    # agent's own current semantic state
        agent_feeling:      np.ndarray | None = None,  # [valence, arousal] of agent
        plasticity_rate:    float = 1.0,
    ) -> dict:
        """Update the interlocutor belief state from one observed utterance.

        Parameters
        ----------
        observed_semantic : np.ndarray [semantic_dim]
            Semantic embedding of what the interlocutor just said
            (from WernickeModule processing their text).
        agent_semantic : np.ndarray [semantic_dim]
            The agent's own current semantic vector (used to estimate
            knowledge gap and to train the mentalising weights).
        agent_feeling : np.ndarray | None
            Agent's [valence, arousal].  Used as prior for simulated_affect
            (simulation theory — we simulate the other using our own affect).
        plasticity_rate : float

        Returns
        -------
        dict with keys:
            interlocutor_topic   : np.ndarray [semantic_dim]
            knowledge_gap        : float  — mean absolute difference
            simulated_affect     : np.ndarray [2]  — [valence, arousal]
            mentalising_error    : float
            shared_ground        : float  — cosine similarity of topics
        """
        obs = _fit(observed_semantic, self.semantic_dim)
        agt = _fit(agent_semantic,    self.semantic_dim)

        # ----------------------------------------------------------------
        # 1. Update interlocutor topic (exponential smoothing, α=0.3)
        # ----------------------------------------------------------------
        self.interlocutor_topic = 0.7 * self.interlocutor_topic + 0.3 * obs
        self._history.append(obs.copy())

        # ----------------------------------------------------------------
        # 2. Mentalising prediction error
        #    Predict what the interlocutor would say from the agent's state
        # ----------------------------------------------------------------
        predicted_interlocutor = self._W_mentalise @ agt
        error_vec = obs - predicted_interlocutor
        self.mentalising_error = float(np.linalg.norm(error_vec))

        # Hebbian update: reduce prediction error (TPJ-mediated learning)
        eta = 0.002 * plasticity_rate
        dW  = eta * np.outer(error_vec, agt)
        self._W_mentalise = np.clip(self._W_mentalise + dW, -1.0, 1.0)

        # ----------------------------------------------------------------
        # 3. Knowledge gap estimate
        #    How different is their semantic history from the agent's current topic?
        # ----------------------------------------------------------------
        if len(self._history) >= 2:
            other_mean = np.mean(list(self._history), axis=0)
            self.belief_about_other_knowledge = (
                0.9 * self.belief_about_other_knowledge + 0.1 * other_mean)
        gap = float(np.mean(np.abs(agt - self.belief_about_other_knowledge)))

        # ----------------------------------------------------------------
        # 4. Simulated affect (simulation theory of mind)
        #    Use the agent's own affective sensitivity to estimate how
        #    the interlocutor likely feels about the observed topic.
        # ----------------------------------------------------------------
        raw_affect = self._W_affect @ obs     # [2]
        raw_affect[0] = np.tanh(raw_affect[0])              # valence ∈ (-1, 1)
        raw_affect[1] = float(np.clip(raw_affect[1], 0, 1)) # arousal ∈ (0, 1)

        pred_affect = raw_affect.copy()

        # Blend with agent's own affect as prior (simulation theory)
        if agent_feeling is not None:
            af = _fit(agent_feeling, 2)
            raw_affect = 0.6 * raw_affect + 0.4 * af

        self.simulated_affect = raw_affect.astype(np.float32)

        # Update affect weights from agent's own ground-truth feeling
        if agent_feeling is not None:
            af2 = _fit(agent_feeling, 2)
            dW_aff = eta * np.outer(af2 - pred_affect, obs)
            self._W_affect = np.clip(self._W_affect + dW_aff, -1.0, 1.0)

        # ----------------------------------------------------------------
        # 5. Shared common ground (cosine similarity)
        # ----------------------------------------------------------------
        shared = float(np.dot(agt, obs) / (
            np.linalg.norm(agt) * np.linalg.norm(obs) + 1e-8))

        return {
            'interlocutor_topic':  self.interlocutor_topic,
            'knowledge_gap':       gap,
            'simulated_affect':    self.simulated_affect,
            'mentalising_error':   self.mentalising_error,
            'shared_ground':       shared,
        }

    def should_explain(self, threshold: float = 0.4) -> bool:
        """Return True if knowledge gap suggests the interlocutor needs explanation."""
        gap = float(np.mean(np.abs(
            self.interlocutor_topic - self.belief_about_other_knowledge)))
        return gap > threshold

    def reset(self) -> None:
        """Reset at the start of a new dialogue session."""
        self.interlocutor_topic[:] = 0.0
        self.belief_about_other_knowledge[:] = 0.5
        self.simulated_affect[:] = [0.0, 0.5]
        self._history.clear()
        self.mentalising_error = 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate 1-D float32 array to length n."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))
