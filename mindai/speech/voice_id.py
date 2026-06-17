"""VoiceID — deterministic voice fingerprint from a brain seed.

Solves the "uniqueness problem": a brain that listens to many speakers
would otherwise mimic them all and never have a stable voice. We fix
the synthesiser parameters at brain birth using a single integer seed
(saved with the brain weights).

Parameters chosen deterministically from the seed:
    base_voice    — index into a curated list of TTS voices
    pitch_shift   — [-3, +3] semitones from base
    rate          — [0.85, 1.15] speed multiplier
    timbre_warm   — [0.0, 1.0] formant emphasis (warmth)

These define the brain's VOCAL APPARATUS (the physical body of the voice).
What the brain SAYS is determined by motor neurons; HOW it sounds is the
fixed VoiceID.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# Curated set of base voices known to work with edge-tts (Microsoft Edge TTS).
# Mix of male/female, multiple accents — gives 8 distinct voice "bodies".
_BASE_VOICES = [
    'en-US-AriaNeural',         # female, warm
    'en-US-GuyNeural',          # male, neutral
    'en-US-JennyNeural',        # female, natural
    'en-US-DavisNeural',        # male, expressive
    'en-GB-SoniaNeural',        # female, British
    'en-GB-RyanNeural',         # male, British
    'en-AU-NatashaNeural',      # female, Australian
    'en-IE-EmilyNeural',        # female, Irish
]


def list_voices() -> list[dict]:
    """All selectable voices with friendly metadata for the GUI."""
    meta = {
        'en-US-AriaNeural':     ('Aria',    'US',        'female'),
        'en-US-GuyNeural':      ('Guy',     'US',        'male'),
        'en-US-JennyNeural':    ('Jenny',   'US',        'female'),
        'en-US-DavisNeural':    ('Davis',   'US',        'male'),
        'en-GB-SoniaNeural':    ('Sonia',   'British',   'female'),
        'en-GB-RyanNeural':     ('Ryan',    'British',   'male'),
        'en-AU-NatashaNeural':  ('Natasha', 'Australian','female'),
        'en-IE-EmilyNeural':    ('Emily',   'Irish',     'female'),
    }
    return [
        {'id': v, 'name': meta[v][0], 'accent': meta[v][1], 'gender': meta[v][2]}
        for v in _BASE_VOICES
    ]


class VoiceID:
    """Deterministic voice fingerprint derived from a stable seed."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        h = hashlib.sha256(str(self.seed).encode()).digest()

        # Pick base voice from first byte
        self.base_voice = _BASE_VOICES[h[0] % len(_BASE_VOICES)]
        # Pitch shift in [-3, +3] semitones from second byte
        self.pitch_shift = (h[1] / 255.0 - 0.5) * 6.0
        # Rate in [0.85, 1.15] from third byte
        self.rate = 0.85 + (h[2] / 255.0) * 0.30
        # Timbre warmth [0, 1] from fourth byte (currently informational)
        self.timbre_warm = h[3] / 255.0

    # ------------------------------------------------------------------
    # SSML formatting helpers for edge-tts
    # ------------------------------------------------------------------

    @property
    def edge_tts_pitch(self) -> str:
        sign = '+' if self.pitch_shift >= 0 else ''
        return f'{sign}{self.pitch_shift:.1f}st'

    @property
    def edge_tts_rate(self) -> str:
        delta_pct = int((self.rate - 1.0) * 100)
        sign = '+' if delta_pct >= 0 else ''
        return f'{sign}{delta_pct}%'

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'seed':        self.seed,
            'base_voice':  self.base_voice,
            'pitch_shift': self.pitch_shift,
            'rate':        self.rate,
            'timbre_warm': self.timbre_warm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'VoiceID':
        # Re-derive everything from seed for consistency
        return cls(seed=d['seed'])

    @classmethod
    def load_or_create(cls, save_dir: str | Path, seed: int | None = None) -> 'VoiceID':
        """Load voice from save_dir/voice.json, or create with given seed.

        On load, an explicit `override` block (added by user voice picks) takes
        precedence over the seed-derived defaults — so manual overrides survive.
        """
        p = Path(save_dir) / 'voice.json'
        if p.exists():
            d = json.loads(p.read_text())
            v = cls(seed=d['seed'])
            ov = d.get('override') or {}
            if ov.get('base_voice')   in _BASE_VOICES: v.base_voice  = ov['base_voice']
            if 'pitch_shift' in ov:                    v.pitch_shift = float(ov['pitch_shift'])
            if 'rate'        in ov:                    v.rate        = float(ov['rate'])
            return v
        import time
        sd = seed if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
        v = cls(seed=sd)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(v.to_dict(), indent=2))
        return v

    def save_override(self, save_dir: str | Path) -> None:
        """Persist current voice settings as an override on top of the seed."""
        p = Path(save_dir) / 'voice.json'
        base = json.loads(p.read_text()) if p.exists() else {'seed': self.seed}
        base['override'] = {
            'base_voice':  self.base_voice,
            'pitch_shift': self.pitch_shift,
            'rate':        self.rate,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(base, indent=2))

    def __repr__(self) -> str:
        return (f'VoiceID(seed={self.seed}, voice={self.base_voice}, '
                f'pitch={self.pitch_shift:+.1f}Hz, rate={self.rate:.2f}x)')
