"""CharTokenizer — Character-level tokenizer for Russian and English.

Naturally maps Russian and English characters to unique token IDs.
"""

from __future__ import annotations


class CharTokenizer:
    """Character-level tokenizer for Russian and English."""

    def __init__(self):
        # We define a fixed list of characters covering both Russian and English
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "0123456789"
            " \n\t\r"
            ".,!?;:-_+=@#$%^&*()[]{}<>/\\|'\"~`"
        )
        # Deduplicate while preserving order (should be already unique, but safe)
        seen = set()
        self._chars = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                self._chars.append(c)

        self._char_to_id = {c: i for i, c in enumerate(self._chars)}
        
        # Define unk and eos ids after the normal character ids
        self.unk_id = len(self._chars)
        self.eos_id = self.unk_id + 1
        self.vocab_size = self.eos_id + 1

    def encode(self, text: str) -> list[int]:
        ids = []
        for c in text:
            ids.append(self._char_to_id.get(c, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i < len(self._chars):
                chars.append(self._chars[i])
        return "".join(chars)


def get_tokenizer(name: str = 'auto', **kwargs) -> CharTokenizer:
    """Return a CharTokenizer. `name` is accepted for backward compatibility."""
    return CharTokenizer()
