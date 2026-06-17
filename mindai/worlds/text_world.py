"""TextWorld — text token stream as the sensory environment.

Replaces 3D/2D world with a stream of tokenized text.
The brain receives token patterns as sensory input and generates
token patterns as motor output — pure language prediction via STDP
and predictive coding, no scripted reward.

Biological analogy:
  The infant brain receives phoneme streams via auditory cortex.
  Statistical regularities in the stream drive STDP — the brain
  learns to predict the next phoneme before the motor system learns
  to produce speech.  Here we skip the acoustic level and work
  directly with subword tokens (BPE), which is an abstraction but
  preserves the sequence-prediction structure.

Homeostatic signals:
  No hunger/thirst — this brain has no body.
  Curiosity (novelty): high when prediction error (surprise) is high.
  Fatigue: rises with tick count, reset by "sleep" (save+reload).
  Pain proxy: used to mark malformed / out-of-distribution tokens.
"""

from __future__ import annotations

import random
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from mindai.worlds.tokenizers import get_tokenizer


# ---------------------------------------------------------------------------
# TextWorld
# ---------------------------------------------------------------------------

class TextWorld:
    """Feeds a token stream as sensory input to Brain.

    Interface identical to GridWorld so Brain.run() works unchanged.

    Sensory layout used in main_text.py:
        'token_in'  : TOKEN_NEURONS  — current token pattern
        'token_ctx' : TOKEN_NEURONS  — previous token (context)
        'curiosity' : 10             — novelty / surprise signal
        'fatigue'   : 10             — tiredness proxy
    Motor layout:
        'motor'     : TOKEN_NEURONS  — decoded as output token
    """

    # Neurons per token pattern (one slot).
    # Two slots are concatenated into the vision channel:
    #   raw[0 : TOKEN_NEURONS]           = current token
    #   raw[TOKEN_NEURONS : 2*TOKEN_NEURONS] = previous token (context)
    # Total vision channel size = 2 * TOKEN_NEURONS.
    TOKEN_NEURONS = 2048

    def __init__(
        self,
        corpus_path: str | None = None,
        vocab_size:  int = 512,
        max_context: int = 64,
        stream_interactive: bool = True,
    ):
        self.tokenizer    = get_tokenizer()
        self.vocab_size   = min(vocab_size, self.tokenizer.vocab_size)
        self.max_context  = max_context
        self.stream_interactive = stream_interactive

        # Build fixed random sparse patterns for each token.
        # Pattern is a unit vector in TOKEN_NEURONS-dimensional space.
        # Fixed seed so patterns are consistent across runs.
        rng = np.random.default_rng(seed=42)
        self._patterns = np.zeros(
            (self.tokenizer.vocab_size, self.TOKEN_NEURONS), dtype=np.float32)
        active_per_token = max(4, self.TOKEN_NEURONS // 50)   # ~2% active
        for tid in range(self.tokenizer.vocab_size):
            active_idx = rng.choice(self.TOKEN_NEURONS, active_per_token, replace=False)
            self._patterns[tid, active_idx] = 1.0

        # Internal state
        self._tick:          int   = 0
        self._curiosity:     float = 0.5
        self._motor_pattern: np.ndarray = np.zeros(self.TOKEN_NEURONS, dtype=np.float32)
        self._fatigue:       float = 0.0
        self._alive:         bool  = True
        self._context:       deque[int] = deque(maxlen=max_context)   # recent token ids
        self._current_token: int   = 0
        self._next_token:    int   = 0
        self._surprise_accum: float = 0.0

        # Corpus iterator
        self._corpus_lines: list[str] = []
        self._corpus_idx:   int = 0
        if corpus_path and Path(corpus_path).exists():
            with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                self._corpus_lines = [l.strip() for l in f if l.strip()]
            print(f'>>> TextWorld: загружен корпус {len(self._corpus_lines)} строк')
        else:
            # Built-in seed corpus — minimal English for bootstrapping
            self._corpus_lines = _SEED_CORPUS
            print('>>> TextWorld: встроенный seed-корпус')

        # Interactive I/O buffers
        self._input_queue:  deque[int] = deque()   # pending input tokens
        self._output_ids:   list[int] = []   # generated output token ids
        self._output_text:  str = ''

        # World sound / vocalization stubs (Brain.run() expects these)
        self._sound_queue: deque[np.ndarray] = deque()
        self.last_agent_vocalization = np.zeros(self.TOKEN_NEURONS, dtype=np.float32)
        self.isolation_ticks = 0
        self.agent_pos   = [0, 0]
        self.human_pos   = [0, 0]
        self.world_tick  = 0

        self._load_next_token()

    # ------------------------------------------------------------------
    # Token loading
    # ------------------------------------------------------------------

    def _load_next_token(self) -> None:
        """Advance one token from corpus or input queue."""
        if self._input_queue:
            self._current_token = self._next_token
            self._next_token    = self._input_queue.popleft()
            return

        # Advance in corpus
        while True:
            if self._corpus_idx >= len(self._corpus_lines):
                self._corpus_idx = 0
                random.shuffle(self._corpus_lines)
            line = self._corpus_lines[self._corpus_idx]
            self._corpus_idx += 1
            ids = self.tokenizer.encode(line)
            if ids:
                self._corpus_tokens = ids
                self._corpus_token_idx = 0
                break

        if not hasattr(self, '_corpus_tokens') or self._corpus_token_idx >= len(self._corpus_tokens):
            self._corpus_tokens = self.tokenizer.encode(
                self._corpus_lines[self._corpus_idx % len(self._corpus_lines)])
            self._corpus_token_idx = 0

        self._current_token = self._next_token
        self._next_token    = self._corpus_tokens[self._corpus_token_idx]
        self._corpus_token_idx += 1

    # ------------------------------------------------------------------
    # Brain.run() interface
    # ------------------------------------------------------------------

    def get_homeostatic_signals(self) -> dict[str, float]:
        # Brain in a jar: no body, no homeostatic pressure.
        # Curiosity emerges from prediction error (PredictiveMicrocircuits → ACh/DA).
        # Fatigue accumulates via adenosine (CircadianRhythm).
        # Both are internal brain states — not external electrode signals.
        return {
            'pain':   0.0,
            'hunger': 0.0,
            'thirst': 0.0,
        }

    def get_sensory_retina(self, num_neurons: int) -> dict[str, np.ndarray]:
        """Return current + context token patterns as dict."""
        cur_pat = self._patterns[self._current_token % len(self._patterns)]
        ctx_pat = np.zeros(self.TOKEN_NEURONS, dtype=np.float32)
        if self._context:
            ctx_pat = self._patterns[self._context[-1] % len(self._patterns)] * 0.7
        return {
            'token_in':  cur_pat,
            'token_ctx': ctx_pat,
        }

    def receive_motor_pattern(self, motor_signals: np.ndarray) -> None:
        """Receive raw motor neuron activity from brain.py each tick.

        Stored and used in execute_action for nearest-neighbour token decoding.
        This bypasses BasalGanglia action-index discretisation — the motor
        pattern IS the speech output, decoded by similarity to known token
        embeddings.  Analogous to motor cortex → articulators → phonemes.
        """
        n = min(len(motor_signals), self.TOKEN_NEURONS)
        self._motor_pattern[:n] = motor_signals[:n]
        if n < self.TOKEN_NEURONS:
            self._motor_pattern[n:] = 0.0

    def _decode_motor_to_token(self) -> int:
        """Nearest-neighbour: find token whose pattern best matches motor activity.

        Dot product between motor pattern and each token pattern.
        Highest dot product = most similar population code = predicted token.
        This is how motor cortex population vectors are decoded in BCI research
        (Georgopoulos 1986 population vector coding).
        """
        # _patterns: (vocab_size, TOKEN_NEURONS)
        # dot products: (vocab_size,)
        scores = self._patterns @ self._motor_pattern
        return int(np.argmax(scores))

    def execute_action(self, motor_idx: int) -> dict:
        """Decode motor pattern as token prediction; compare to actual next token.

        Uses nearest-neighbour pattern matching if motor pattern was received
        this tick (receive_motor_pattern called), otherwise falls back to
        motor_idx from BasalGanglia (GridWorld compatibility).
        """
        motor_norm = float(np.linalg.norm(self._motor_pattern))
        if motor_norm > 0.01:
            predicted_token = self._decode_motor_to_token()
        else:
            predicted_token = motor_idx % self.tokenizer.vocab_size

        # Advance corpus
        self._context.append(self._current_token)

        actual_next = self._next_token
        correct = (predicted_token == actual_next)

        # Curiosity: high when prediction wrong; low when right (Berlyne 1960)
        if correct:
            self._curiosity = max(0.1, self._curiosity * 0.95)
            energy_signal   = 0.3
        else:
            self._curiosity = min(1.0, self._curiosity + 0.05)
            energy_signal   = 0.0

        # Collect generated text
        self._output_ids.append(predicted_token)
        if len(self._output_ids) > 200:
            self._output_ids = self._output_ids[-200:]
        self._output_text = self.tokenizer.decode(self._output_ids[-80:])

        self._load_next_token()
        self._tick += 1
        self.world_tick += 1

        # Fatigue accumulates slowly
        self._fatigue = min(1.0, self._fatigue + 0.00005)

        return {
            'energy': energy_signal,
            'water':  0.0,
            'stress': 0.0 if correct else 0.1,
        }

    def is_alive(self) -> bool:
        return self._alive

    def process_human_input(self, keys: dict) -> None:
        """Accept typed text input from UI or stdin."""
        text = keys.get('text_input', '')
        if text:
            ids = self.tokenizer.encode(text)
            self._input_queue.extend(ids)

    def get_distance_to_human(self) -> float:
        return float('inf')   # no human in text world

    def pop_world_sound(self) -> np.ndarray:
        if self._sound_queue:
            return self._sound_queue.popleft()
        return np.zeros(32, dtype=np.float32)

    def add_sound(self, pos, sound: np.ndarray) -> None:
        pass

    def receive_vocalization(self, vocal: np.ndarray) -> None:
        self.last_agent_vocalization = vocal

    def get_current_output(self) -> str:
        return self._output_text

    def inject_prompt(self, text: str) -> None:
        """Feed a text prompt directly into the token queue."""
        ids = self.tokenizer.encode(text)
        self._input_queue.extend(ids)
        print(f'>>> Промпт: {len(ids)} токенов в очереди')


# ---------------------------------------------------------------------------
# Built-in seed corpus
# ---------------------------------------------------------------------------

_SEED_CORPUS = [
    "The brain is a prediction machine.",
    "Neurons that fire together wire together.",
    "Learning emerges from experience.",
    "The sky is blue and the grass is green.",
    "Language is a sequence of symbols.",
    "Intelligence is the ability to adapt.",
    "Memory is stored in synaptic weights.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello, how are you today?",
    "I am learning to understand language.",
    "What is the meaning of consciousness?",
    "The sun rises in the east and sets in the west.",
    "Numbers: one two three four five six seven eight nine ten.",
    "Colors: red orange yellow green blue indigo violet.",
    "Time passes and patterns emerge from repetition.",
    "To be or not to be, that is the question.",
    "The answer to life, the universe, and everything is forty two.",
    "Water flows downhill following the path of least resistance.",
    "Fire needs oxygen, fuel, and heat to burn.",
    "Plants convert sunlight into energy through photosynthesis.",
    "The heart pumps blood through the body.",
    "The brain contains approximately eighty six billion neurons.",
    "Synapses are the connections between neurons.",
    "Sleep is essential for memory consolidation.",
    "Dreams occur during REM sleep.",
    "Dopamine is released when we experience rewards.",
    "Fear is processed by the amygdala.",
    "Language is primarily a left hemisphere function.",
    "Vision is processed in the occipital lobe.",
    "Movement is controlled by the motor cortex.",
    "I think therefore I am.",
    "Knowledge is power.",
    "Practice makes perfect.",
    "Every action has an equal and opposite reaction.",
    "Energy cannot be created or destroyed only transformed.",
]
