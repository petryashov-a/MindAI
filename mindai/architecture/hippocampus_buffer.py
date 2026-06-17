import heapq
import numpy as np
from collections import deque

class Hippocampus:

    def __init__(self, max_capacity: int=1000):
        self.max_capacity = max_capacity
        self.episodic_memory = deque(maxlen=max_capacity)
        self.current_time_index = 0

    def encode_episode(self, workspace_pattern: np.ndarray, emotional_valence: float):
        self.current_time_index += 1
        if abs(emotional_valence) < 0.2:
            return
        episode = {
            'id':        self.current_time_index,
            'timestamp': self.current_time_index,
            'pattern':   workspace_pattern.copy(),
            'valence':   emotional_valence,
        }
        self.episodic_memory.append(episode)

    def retrieve_for_nrem(self) -> list:
        """Age-decay + top-50 by |valence| desc (Stickgold 2005). Cortisol not applied —
        cortisol is at nadir during N3, so mutation would be biologically wrong.
        Uses heapq.nlargest — O(n log k) instead of O(n log n)."""
        result = []
        for ep in self.episodic_memory:
            decay = float(np.exp(-(self.current_time_index - ep['timestamp']) / 1000.0))
            result.append({
                'id':      ep['id'],
                'pattern': ep['pattern'].copy() * decay,
                'valence': ep['valence'],
            })
        return heapq.nlargest(min(50, len(result)), result, key=lambda e: abs(e['valence']))

    def retrieve_for_rem(self) -> list:
        """High-valence episodes (|valence| > 0.4) for REM emotional reprocessing.
        Returns references to originals so reduce_valence() can update in-place."""
        return [ep for ep in self.episodic_memory if abs(ep['valence']) > 0.4]

    def reduce_valence(self, episode_id: int, factor: float = 0.85) -> None:
        """Strip emotional charge after REM replay with NA=0 (Walker 2009).
        Memory content preserved; only valence magnitude shrinks."""
        for ep in self.episodic_memory:
            if ep['id'] == episode_id:
                ep['valence'] *= factor
                return

    def retrieve_for_consolidation(self, current_cortisol: float, current_mood_vector: np.ndarray) -> list:
        """Legacy method — kept for backward compat (surgery.py etc.)."""
        retrieved_patterns = []
        rng = np.random.default_rng(42)
        for ep in self.episodic_memory:
            pattern = ep['pattern'].copy()
            age = self.current_time_index - ep['timestamp']
            decay = np.exp(-age / 1000.0)
            pattern *= decay
            if current_cortisol > 0.3:
                flip_probability = (current_cortisol - 0.3) * 0.2
                mutation_mask = rng.random(pattern.shape) < flip_probability
                pattern = np.where(mutation_mask, 1.0 - pattern, pattern)
            # Safe shape broadcast
            mood = current_mood_vector[:pattern.shape[0]]
            if len(mood) < len(pattern):
                mood = np.pad(mood, (0, len(pattern) - len(mood)))
            pattern = pattern * 0.8 + mood * 0.2
            retrieved_patterns.append(pattern)
        return retrieved_patterns