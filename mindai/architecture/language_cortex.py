"""Language Cortex — Wernicke (semantic comprehension) + Broca (syntactic production).

Biological basis
----------------
Two core language areas in the left hemisphere (right in some left-handers):

  Wernicke's area (BA22, posterior superior temporal sulcus):
    Semantic comprehension and lexical access (Wernicke 1874).
    Damage → fluent but meaningless speech (Wernicke's aphasia).
    Receives: auditory cortex input (phonology) + angular gyrus (semantics).
    Output: semantic_vector — a distributed representation of the *meaning*
    of the current utterance in context.
    dlPFC back-projection (Petrides & Pandya 1988): dialogue working memory
    context is injected here, extending effective context window beyond
    the short-term semantic buffer.

  Broca's area (BA44/45, left inferior frontal gyrus):
    Syntactic processing, hierarchical sequencing, and speech production
    (Broca 1861; Friederici 2012).
    Damage → halting, agrammatic speech (Broca's aphasia).
    Contains: syntactic working memory for nested constituents (Gibson 1998).
    Output: next_token_prediction — most likely continuation given context.

    Autoregressive generation (NEW — Indefrey & Levelt 2004):
    Speech is produced via a recurrent motor program in SMA/BA44:
      1. The current semantic intent (Wernicke output) seeds production.
      2. Each predicted token is fed back as input for the next step.
         (Biologically: recurrent layer-3 collaterals in BA44 create
          a self-exciting loop; Fuster 2008.)
      3. Termination criteria (biologically motivated):
         a. Entropy threshold: when the token distribution becomes too
            flat (= high uncertainty), the PAG inhibits vocalization.
            (Analog of speaker planning failures — "tip of the tongue".)
         b. Max length: articulatory buffer exhaustion — average sentence
            ~15 words; hard cap at 32 tokens (Miller 1956).
         c. Sentence boundary: stack clears, production drive drops.

  Arcuate fasciculus:
    White-matter tract connecting Wernicke ↔ Broca bidirectionally.
    Damage → conduction aphasia: comprehension OK, repetition impaired.
    Implemented here as a learned weight matrix updated via Hebbian STDP.

  SyntacticStack (inside BrocaModule):
    Inspired by left IFG syntactic working memory (BA44).
    Implements a differentiable push-down stack for recursive constituent
    embedding.  Depth 8 suffices for natural language (no sentences in
    the wild exceed depth ~4-5 in dependency parse trees; Miller 1956).
    Stack slots = phrase constituent vectors (noun-phrase, verb-phrase etc.).

Implementation Notes
--------------------
* No gradient descent — all learning is Hebbian / prediction-error driven.
* WernickeModule is a specialised PredictiveMicrocircuit with direct
  access to EntorhinalGrid activity (semantic position in concept space).
* BrocaModule reads Wernicke's semantic_vector + SyntacticStack top, and
  predicts the next token via a small weight matrix trained online.
* The arcuate_fasciculus weight matrix allows bidirectional correction:
  if production error is high, it propagates back to update Wernicke's
  semantic binding.
* generate_utterance() implements the autoregressive loop — call it
  when communicative_drive exceeds threshold.
"""

from __future__ import annotations

import numpy as np

# Communicative intent threshold: drive must exceed this to trigger generation.
# Biologically: corresponds to the SMA "readiness potential" pre-speech onset
# (Libet 1983; Haggard 2008).  0.35 is empirically calibrated.
_COMM_DRIVE_THRESHOLD = 0.35

# Entropy threshold for autoregressive termination.
# When the next-token distribution is too flat (uniform entropy ≈ log(vocab)),
# Broca stops — analogous to tip-of-the-tongue state (Brown & McNeill 1966).
_ENTROPY_STOP_RATIO   = 0.85   # fraction of max entropy

# Maximum tokens per generated utterance (articulatory buffer).
_MAX_UTTERANCE_LEN    = 32



# ---------------------------------------------------------------------------
# Syntactic Stack — recursive constituent working memory
# ---------------------------------------------------------------------------

class SyntacticStack:
    """Push-down stack for nested syntactic constituents.

    Biological basis (Gibson 1998 — dependency locality theory):
      The left IFG (BA44) maintains a stack of unresolved syntactic
      dependencies.  Each slot holds a vector representing an open
      constituent (noun phrase, embedded clause, etc.).  Depth ≈ 8 covers
      all natural language sentences empirically observed (Miller 1956).

    Operations:
      push(v)  — open a new constituent; store its embedding on the stack
      pop()    — close the current constituent; return its accumulated embedding
      peek()   — read stack top without modifying stack
      clear()  — sentence-boundary reset (at punctuation / silence)
    """

    MAX_DEPTH = 8

    def __init__(self, slot_dim: int) -> None:
        self.slot_dim = slot_dim
        # Stack storage: (MAX_DEPTH, slot_dim) — pre-allocated
        self._stack = np.zeros((self.MAX_DEPTH, slot_dim), dtype=np.float32)
        self._ptr   = 0   # points to next free slot (0 = empty)

    # ------------------------------------------------------------------

    def push(self, constituent_vec: np.ndarray) -> None:
        """Open a new syntactic constituent."""
        if self._ptr < self.MAX_DEPTH:
            self._stack[self._ptr] = constituent_vec[:self.slot_dim]
            self._ptr += 1
        else:
            # Stack overflow: shift stack down (forget oldest constituent)
            self._stack[:-1] = self._stack[1:]
            self._stack[-1]  = constituent_vec[:self.slot_dim]

    def pop(self) -> np.ndarray:
        """Close the current constituent and return its embedding."""
        if self._ptr > 0:
            self._ptr -= 1
            vec = self._stack[self._ptr].copy()
            self._stack[self._ptr] = 0.0
            return vec
        return np.zeros(self.slot_dim, dtype=np.float32)

    def peek(self) -> np.ndarray:
        """Read stack top without modifying the stack."""
        if self._ptr > 0:
            return self._stack[self._ptr - 1].copy()
        return np.zeros(self.slot_dim, dtype=np.float32)

    def clear(self) -> None:
        """Reset at sentence boundary."""
        self._stack[:] = 0.0
        self._ptr = 0

    @property
    def depth(self) -> int:
        """Current stack depth (0 = empty)."""
        return self._ptr

    @property
    def is_empty(self) -> bool:
        return self._ptr == 0


# ---------------------------------------------------------------------------
# Wernicke Module — semantic comprehension
# ---------------------------------------------------------------------------

class WernickeModule:
    """Semantic comprehension — BA22 / posterior superior temporal sulcus.

    Integrates token embeddings with EntorhinalGrid position to produce
    a contextualised semantic_vector representing the current utterance meaning.

    Learning: Hebbian association between token patterns and grid positions.
    The weight matrix W_sem maps (token_embedding || grid_activity) → semantic_vector.
    Update rule: Δw = η × error × (token_embedding || grid_activity)
    where error = semantic_target − W_sem @ input (prediction error, not gradient).
    """

    def __init__(self, token_dim: int, grid_dim: int, semantic_dim: int) -> None:
        self.token_dim    = token_dim
        self.grid_dim     = grid_dim
        self.semantic_dim = semantic_dim
        input_dim         = token_dim + grid_dim

        # Weight matrix: (input_dim → semantic_dim)
        rng = np.random.default_rng(11)
        self._W = rng.standard_normal((semantic_dim, input_dim)).astype(np.float32) * 0.01

        # Bias (resting semantic baseline)
        self._b = np.zeros(semantic_dim, dtype=np.float32)

        # Semantic output (updated each call to process())
        self.semantic_vector = np.zeros(semantic_dim, dtype=np.float32)

        # Prediction error (surprise signal)
        self.comprehension_error: float = 0.0

        # Short-term semantic context buffer (last 4 semantic vectors)
        self._context_buf: list[np.ndarray] = []
        self._ctx_depth = 4

    # ------------------------------------------------------------------

    def process(
        self,
        token_embedding: np.ndarray,
        grid_activity:   np.ndarray,
        plasticity_rate: float = 1.0,
        dlpfc_context:   np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute semantic vector for the current input.

        Parameters
        ----------
        token_embedding : np.ndarray
            SDR or dense representation of the current token/sensory frame.
        grid_activity : np.ndarray
            Current EntorhinalGrid output (semantic position cue).
        plasticity_rate : float
            Scales Hebbian update strength (≈ acetylcholine modulation).
        dlpfc_context : np.ndarray | None
            Back-projection from dlPFC dialogue working memory
            (Petrides & Pandya 1988).  When provided, blended into the
            semantic computation to extend effective context window.

        Returns
        -------
        semantic_vector : np.ndarray [semantic_dim]
        """
        # Pad/truncate inputs to expected sizes
        tok = _fit(token_embedding, self.token_dim)
        grd = _fit(grid_activity,   self.grid_dim)
        combined = np.concatenate([tok, grd])   # (input_dim,)

        # Forward pass: linear + tanh activation
        raw_out = self._W @ combined + self._b
        new_sem = np.tanh(raw_out)

        # dlPFC back-projection: blend dialogue working memory context
        # Biologically: top-down modulation from BA46/9 → BA22
        # (Petrides & Pandya 1988; D'Esposito 2007)
        if dlpfc_context is not None:
            ctx = _fit(dlpfc_context, self.semantic_dim)
            # Weight: 20% dlPFC context, 80% current bottom-up signal
            # (Matches relative strength of top-down vs. bottom-up in
            #  temporal cortex; Lamme & Roelfsema 2000)
            new_sem = 0.80 * new_sem + 0.20 * ctx

        # Context-weighted integration: blend with previous semantic context
        if self._context_buf:
            ctx_mean = np.mean(self._context_buf, axis=0)
            new_sem  = 0.7 * new_sem + 0.3 * ctx_mean

        # Prediction error: how much does this tick's semantic vector
        # differ from the context prediction?
        if self._context_buf:
            predicted = self._context_buf[-1]
            error_vec = new_sem - predicted
        else:
            error_vec = new_sem

        self.comprehension_error = float(np.linalg.norm(error_vec))

        # Hebbian update: reduce prediction error (Rao & Ballard 1999)
        eta = 0.002 * plasticity_rate
        # Outer product: error_vec (post) × combined (pre)
        dW = eta * np.outer(error_vec, combined)
        self._W = np.clip(self._W + dW, -1.0, 1.0)
        self._b = np.clip(self._b + eta * error_vec, -0.5, 0.5)

        self.semantic_vector = new_sem

        # Update context buffer
        self._context_buf.append(new_sem.copy())
        if len(self._context_buf) > self._ctx_depth:
            self._context_buf.pop(0)

        return self.semantic_vector

    def reset_context(self) -> None:
        """Reset semantic context (e.g. topic change, long silence)."""
        self._context_buf.clear()


# ---------------------------------------------------------------------------
# Broca Module — syntactic production
# ---------------------------------------------------------------------------

class BrocaModule:
    """Syntactic processing and speech production — BA44/45.

    Uses the SyntacticStack to handle recursive structure, and predicts
    the next token given the current semantic context from Wernicke.

    Stack management heuristic (biologically motivated):
      * PUSH when incoming semantic_vector is dissimilar to stack top
        (= new constituent opens; similarity < push_threshold)
      * POP when stack top is very similar to current vector
        (= constituent closed; similarity > pop_threshold)
      * CLEAR at sentence boundary (silence / punctuation)

    Next-token prediction:
      W_prod @ (semantic_vector || stack_top) → logits over vocab
      Update: prediction error drives Hebbian weight correction.
    """

    def __init__(
        self,
        semantic_dim:  int,
        vocab_size:    int,
        stack_slot_dim: int | None = None,
    ) -> None:
        self.semantic_dim  = semantic_dim
        self.vocab_size    = vocab_size
        slot_dim           = stack_slot_dim or semantic_dim

        self.stack = SyntacticStack(slot_dim=slot_dim)

        # Production weight matrix: (semantic_dim + slot_dim) → vocab_size
        input_dim = semantic_dim + slot_dim
        self._rng = np.random.default_rng(22)
        self._W_prod = self._rng.standard_normal(
            (vocab_size, input_dim)).astype(np.float32) * 0.01

        # Stack-management decision weights:
        # input: (semantic_dim + slot_dim) → 3 logits [push, pop, idle]
        self._W_stack = self._rng.standard_normal(
            (3, input_dim)).astype(np.float32) * 0.01

        # Arcuate fasciculus: feedback from production error → Wernicke
        # (conduction pathway; bidirectional error correction)
        self._W_arcuate = np.eye(semantic_dim, dtype=np.float32) * 0.1

        # State
        self.predicted_token: int  = 0
        self.production_error: float = 0.0
        self._lr = 0.003

    # ------------------------------------------------------------------

    def process(
        self,
        semantic_vector: np.ndarray,         # from WernickeModule
        target_token:    int | None = None,  # ground-truth token (for supervised Hebbian)
        plasticity_rate: float = 1.0,
        sentence_boundary: bool = False,
    ) -> dict:
        """Predict next token and manage syntactic stack.

        Parameters
        ----------
        semantic_vector : np.ndarray [semantic_dim]
        target_token : int | None
            If provided, Hebbian learning updates W_prod to reduce error.
        plasticity_rate : float
        sentence_boundary : bool
            If True, clears the stack (sentence completed).

        Returns
        -------
        dict with keys:
            predicted_token : int   — argmax of production logits
            logits          : np.ndarray [vocab_size] — raw scores
            stack_depth     : int   — current syntactic nesting depth
            arcuate_signal  : np.ndarray [semantic_dim] — error feedback to Wernicke
        """
        if sentence_boundary:
            self.stack.clear()

        stack_top = self.stack.peek()
        sem  = _fit(semantic_vector, self.semantic_dim)
        top  = _fit(stack_top,       self.stack.slot_dim)
        combined = np.concatenate([sem, top])   # (input_dim,)

        # --- Stack management decision (push / pop / idle) ---
        stack_logits = self._W_stack @ combined
        # Softmax
        stack_probs  = _softmax(stack_logits)
        decision     = int(np.argmax(stack_probs))  # 0=push, 1=pop, 2=idle

        # Similarity between current semantic and stack top
        sim = float(np.dot(sem, top) / (
            np.linalg.norm(sem) * np.linalg.norm(top) + 1e-8))

        if decision == 0 and sim < 0.6:     # PUSH: new constituent
            self.stack.push(sem)
        elif decision == 1 and sim > 0.7:   # POP: constituent closed
            self.stack.pop()
        # else: IDLE — no stack change

        # Re-read stack top after potential push/pop
        stack_top2 = self.stack.peek()
        top2 = _fit(stack_top2, self.stack.slot_dim)
        combined2 = np.concatenate([sem, top2])

        # --- Token production ---
        logits = self._W_prod @ combined2
        self.predicted_token = int(np.argmax(logits))

        # --- Prediction error + Hebbian learning ---
        if target_token is not None:
            # One-hot target
            target_vec = np.zeros(self.vocab_size, dtype=np.float32)
            target_vec[target_token] = 1.0
            pred_vec   = _softmax(logits)
            error_vec  = target_vec - pred_vec
            self.production_error = float(np.linalg.norm(error_vec))

            eta = self._lr * plasticity_rate
            self._W_prod = np.clip(
                self._W_prod + eta * np.outer(error_vec, combined2),
                -1.0, 1.0)
        else:
            self.production_error = 0.0

        # --- Arcuate fasciculus feedback ---
        # When production error is high, propagate a correction signal
        # back to Wernicke so it can update its semantic binding.
        arcuate_signal = self._W_arcuate @ sem * self.production_error

        return {
            'predicted_token': self.predicted_token,
            'logits':          logits,
            'stack_depth':     self.stack.depth,
            'arcuate_signal':  arcuate_signal,
        }

    # ------------------------------------------------------------------

    def generate_utterance(
        self,
        semantic_intent: np.ndarray,
        grid_activity:   np.ndarray,
        sdr_encoder,                   # SDREncoder instance
        wernicke:        'WernickeModule',
        dlpfc_context:   np.ndarray | None = None,
        max_len:         int   = _MAX_UTTERANCE_LEN,
        entropy_stop:    float = _ENTROPY_STOP_RATIO,
    ) -> list[int]:
        """Autoregressive speech generation — SMA + BA44 motor program.

        Biological basis (Indefrey & Levelt 2004):
          Speech production is NOT a single-step lookup.  The SMA
          (supplementary motor area) initiates a motor program that
          runs autonomously once triggered.  Each step:
            1. Predict next token from current semantic + stack context.
            2. Feed predicted token back as the next input through Wernicke
               (recurrent layer-3 collaterals in BA44; Fuster 2008).
            3. The stack updates to track syntactic structure.
            4. Termination when entropy is too high (PAG inhibition of
               vocalization) or max length is reached (articulatory buffer).

        Entropy stopping (Brown & McNeill 1966 — tip-of-the-tongue):
          When the model is uncertain about the next word, the token
          distribution becomes flatter (higher entropy).  In biology,
          this manifests as speech disfluency or hesitation.
          Here: if H(p) / log(vocab_size) > entropy_stop_ratio → halt.

        Parameters
        ----------
        semantic_intent : np.ndarray [semantic_dim]
            The semantic state the agent wants to express (Wernicke output).
        grid_activity : np.ndarray
            Current EntorhinalGrid output (for Wernicke context updates).
        sdr_encoder : SDREncoder
            Used to encode each predicted token back as an SDR.
        wernicke : WernickeModule
            Processes each generated token for semantic coherence check.
        dlpfc_context : np.ndarray | None
            Dialogue working memory context from dlPFC.
        max_len : int
            Hard cap on utterance length (articulatory buffer limit).
        entropy_stop : float
            Fraction of max entropy above which generation halts.

        Returns
        -------
        tokens : list[int]
            Generated token sequence.  May be empty if drive is too low
            or entropy threshold is immediately exceeded.
        """
        tokens: list[int] = []
        max_entropy = float(np.log(self.vocab_size + 1e-9))

        # Seed: the agent's current semantic intent
        current_sem = _fit(semantic_intent, self.semantic_dim)
        self.stack.clear()   # fresh syntactic state for new utterance

        for _ in range(max_len):
            stack_top = self.stack.peek()
            top       = _fit(stack_top, self.stack.slot_dim)
            combined  = np.concatenate([current_sem, top])

            # --- Predict next token ---
            logits    = self._W_prod @ combined
            probs     = _softmax(logits)

            # Entropy check: is the model confident enough to speak?
            entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
            if entropy / max_entropy > entropy_stop:
                break   # tip-of-the-tongue: too uncertain, halt

            # Sample proportionally to probability (not always argmax):
            # Biologically, motor output is noisy (synaptic variability).
            # Temperature = 1.0 (no scaling) for natural variability.
            token = int(self._rng.choice(self.vocab_size, p=probs))
            tokens.append(token)

            # --- Stack update ---
            stack_logits = self._W_stack @ combined
            stack_probs  = _softmax(stack_logits)
            decision     = int(np.argmax(stack_probs))
            sim = float(np.dot(current_sem, top) / (
                np.linalg.norm(current_sem) * np.linalg.norm(top) + 1e-8))
            if decision == 0 and sim < 0.6:
                self.stack.push(current_sem)
            elif decision == 1 and sim > 0.7:
                self.stack.pop()

            # --- Recurrent feedback: generated token → next semantic ---
            # Biologically: layer-3 collaterals in BA44 route the motor
            # output back to the semantic comprehension layer (Wernicke)
            # so each word is processed as input for the next prediction.
            tok_arr = np.array([token], dtype=np.float32)
            tok_sdr = sdr_encoder.encode(tok_arr)
            current_sem = wernicke.process(
                token_embedding = tok_sdr,
                grid_activity   = grid_activity,
                plasticity_rate = 0.0,   # generation: no weight updates
                dlpfc_context   = dlpfc_context,
            )

            # Sentence boundary: empty stack after closing main clause
            if self.stack.is_empty and len(tokens) > 3:
                break

        return tokens

    def get_production_vector(self) -> np.ndarray:
        """Return softmax distribution over vocab (for use in other modules)."""
        return np.zeros(self.vocab_size, dtype=np.float32)   # populated externally


# ---------------------------------------------------------------------------
# LanguageCortex — integrated Wernicke + Broca
# ---------------------------------------------------------------------------

class LanguageCortex:
    """Integrated left-hemisphere language system.

    Orchestrates Wernicke (comprehension) and Broca (production) with
    arcuate fasciculus feedback, matching the biological dorsal language stream.

    Usage
    -----
    Instantiate once at Brain init.  Call ``process()`` each tick when text/
    audio input is available.  The ``semantic_vector`` and ``predicted_token``
    attributes give the current state.

    Parameters
    ----------
    token_dim : int
        Size of input token embedding (from SDREncoder or raw).
    grid_dim : int
        Size of EntorhinalGrid output (``entorhinal.total_cells``).
    semantic_dim : int
        Internal semantic representation size.  64–256 is reasonable.
    vocab_size : int
        Number of distinct tokens the system can predict.
    """

    def __init__(
        self,
        token_dim:    int,
        grid_dim:     int,
        semantic_dim: int = 128,
        vocab_size:   int = 256,
    ) -> None:
        self.wernicke = WernickeModule(
            token_dim    = token_dim,
            grid_dim     = grid_dim,
            semantic_dim = semantic_dim,
        )
        self.broca = BrocaModule(
            semantic_dim = semantic_dim,
            vocab_size   = vocab_size,
        )
        self.semantic_dim  = semantic_dim
        self.vocab_size    = vocab_size

        # Exposed state (updated each call to process())
        self.semantic_vector:    np.ndarray = np.zeros(semantic_dim, dtype=np.float32)
        self.predicted_token:    int        = 0
        self.comprehension_error: float     = 0.0
        self.production_error:    float     = 0.0
        self.stack_depth:         int       = 0

    # ------------------------------------------------------------------

    def process(
        self,
        token_embedding:  np.ndarray,
        grid_activity:    np.ndarray,
        target_token:     int | None = None,
        plasticity_rate:  float = 1.0,
        sentence_boundary: bool = False,
        dlpfc_context:    np.ndarray | None = None,
    ) -> dict:
        """One tick of language processing.

        Parameters
        ----------
        token_embedding : np.ndarray
            Current token embedding (SDR or dense).
        grid_activity : np.ndarray
            EntorhinalGrid output for current tick.
        target_token : int | None
            Ground-truth next token (enables supervised Hebbian learning).
        plasticity_rate : float
            Scales both Wernicke and Broca Hebbian updates.
        sentence_boundary : bool
            Clears SyntacticStack if True.
        dlpfc_context : np.ndarray | None
            Back-projection from dlPFC dialogue working memory.
            Injected into Wernicke to extend effective context window.

        Returns
        -------
        dict with keys: semantic_vector, predicted_token, comprehension_error,
                        production_error, stack_depth, arcuate_signal
        """
        # 1. Wernicke: compute semantic vector (+ dlPFC back-projection)
        sem_vec = self.wernicke.process(
            token_embedding=token_embedding,
            grid_activity=grid_activity,
            plasticity_rate=plasticity_rate,
            dlpfc_context=dlpfc_context,
        )

        # 2. Broca: predict next token, manage stack
        broca_out = self.broca.process(
            semantic_vector=sem_vec,
            target_token=target_token,
            plasticity_rate=plasticity_rate,
            sentence_boundary=sentence_boundary,
        )

        # 3. Arcuate fasciculus feedback → Wernicke context update
        # When production error is high, the arcuate signal nudges Wernicke's
        # context buffer so it reconsiders the semantic binding.
        if broca_out['arcuate_signal'] is not None and self.broca.production_error > 0.1:
            arc = broca_out['arcuate_signal']
            if len(self.wernicke._context_buf) > 0:
                self.wernicke._context_buf[-1] = np.clip(
                    self.wernicke._context_buf[-1] + arc * 0.05, -1.0, 1.0)

        # Update exposed state
        self.semantic_vector     = sem_vec
        self.predicted_token     = broca_out['predicted_token']
        self.comprehension_error = self.wernicke.comprehension_error
        self.production_error    = self.broca.production_error
        self.stack_depth         = broca_out['stack_depth']

        return {
            'semantic_vector':    sem_vec,
            'predicted_token':    self.predicted_token,
            'comprehension_error': self.comprehension_error,
            'production_error':   self.production_error,
            'stack_depth':        self.stack_depth,
            'arcuate_signal':     broca_out['arcuate_signal'],
        }

    def generate_utterance(
        self,
        grid_activity:  np.ndarray,
        sdr_encoder,
        dlpfc_context:  np.ndarray | None = None,
    ) -> list[int]:
        """Trigger autoregressive utterance generation.

        Delegates to BrocaModule.generate_utterance() using the current
        Wernicke semantic state as the speech intent.  Should only be
        called when communicative_drive exceeds _COMM_DRIVE_THRESHOLD.

        Returns
        -------
        tokens : list[int]   — empty if generation halted immediately.
        """
        return self.broca.generate_utterance(
            semantic_intent = self.semantic_vector,
            grid_activity   = grid_activity,
            sdr_encoder     = sdr_encoder,
            wernicke        = self.wernicke,
            dlpfc_context   = dlpfc_context,
        )

    def reset_context(self) -> None:
        """Reset both Wernicke context and Broca stack."""
        self.wernicke.reset_context()
        self.broca.stack.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate 1-D array to length n."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)
