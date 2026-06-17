"""GenericGameWorld — run MindAI in any game via window capture + keyboard/mouse.

No mod, no API, no game modification required.

The brain needs only three things:
  1. Vision  — FovealRetina captures any window via mss
  2. Audio   — system audio captured by Cochlea / sounddevice
  3. Motor   — pydirectinput sends DirectInput to any DirectX game

Homeostatic signals (pain, hunger) are optional:
  - Without them: pure STDP learning, no dopamine gating, slower convergence
  - With probes:  user provides Python lambdas that compute signals from the
    raw RGB frame (e.g. sample red health-bar pixel region)

Example (Counter-Strike 2)
--------------------------
    from mindai.worlds.generic import GenericGameWorld

    world = GenericGameWorld(
        window_title = 'Counter-Strike',
        vision_size  = 2880,
        actions = {
            0:  ('key',        'w'),
            1:  ('key',        's'),
            2:  ('key',        'a'),
            3:  ('key',        'd'),
            4:  ('mouse',      'left'),
            5:  ('mouse',      'right'),
            6:  ('key',        'space'),
            7:  ('key',        'shift'),
            8:  ('mouse_move', (15, 0)),
            9:  ('mouse_move', (-15, 0)),
            10: ('mouse_move', (0, 10)),
            11: ('mouse_move', (0, -10)),
        },
        homeostasis = {
            'pain': lambda rgb: _red_bar_probe(rgb, x=20, y=950, w=200, h=10),
        },
    )

Probe helpers are provided at the bottom of this file.
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from mindai.worlds.base import World
from mindai.environment.retina import FovealRetina   # biologically accurate foveal vision
from mindai.worlds.generic.input import GenericInputController


class GenericGameWorld(World):
    """Universal game adapter.

    Parameters
    ----------
    window_title:
        Substring matched against window title (case-insensitive).
    vision_size:
        Number of vision neurons (must be divisible by 5).
    actions:
        Dict mapping motor_idx → action spec tuple.
        See mindai/worlds/generic/input.py for spec formats.
    audio_channels:
        Must match Brain's sensory_layout['audio'].
    homeostasis:
        Optional dict of signal_name → callable(rgb_frame) → float [0,1].
        Called every tick with the raw RGB uint8 frame.
        Keys should match Brain's sensory_layout channels (e.g. 'pain', 'hunger').
    fov_h_deg, fov_v_deg:
        Field of view in degrees (default: human-like 160×130).
    """

    def __init__(
        self,
        window_title:   str,
        vision_size:    int   = 2880,
        actions:        dict  = None,
        audio_channels: int   = 32,
        homeostasis:    Optional[dict[str, Callable]] = None,
        fov_h_deg:      float = 160.0,
        fov_v_deg:      float = 130.0,
    ) -> None:
        self.audio_channels = audio_channels
        self._homeostasis   = homeostasis or {}
        self._last_rgb:     Optional[np.ndarray] = None

        # Default action set: WASD + mouse LMB/RMB + jump + sneak + mouse look
        if actions is None:
            actions = _default_actions()

        self._retina = FovealRetina(
            window_title=window_title,
            vision_size=vision_size,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
        )
        self._input = GenericInputController(actions, audio_channels)

        self._ambient_buffer = np.zeros(audio_channels, dtype=np.float32)
        self._last_agent_vocalization = np.zeros(audio_channels, dtype=np.float32)

        self.world_tick    = 0
        self.inventory     = 'empty'
        self.agent_pos     = [0, 0]
        self.isolation_ticks = 0

        # Release any held keys at start of each tick
        self._input.release_held()

    # ------------------------------------------------------------------
    # World protocol
    # ------------------------------------------------------------------

    def get_sensory_retina(self, num_nodes: int) -> np.ndarray:
        arr = self._retina.get_visual_array()
        # Cache raw RGB for homeostasis probes (avoids double capture)
        self._last_rgb = getattr(self._retina, '_last_raw_rgb', None)
        return arr

    def execute_action(self, motor_idx: int) -> dict:
        self.world_tick += 1
        self._input.release_held()
        self._input.execute(motor_idx)
        return {'energy': 0.0, 'water': 0.0, 'stress': 0.0}

    def get_homeostatic_signals(self) -> dict:
        signals = {}
        if self._homeostasis and self._last_rgb is not None:
            for name, probe in self._homeostasis.items():
                try:
                    val = float(probe(self._last_rgb))
                    signals[name] = float(np.clip(val, 0.0, 1.0))
                except Exception:
                    signals[name] = 0.0
        return signals

    def is_alive(self) -> bool:
        return True   # generic world has no death; user can override

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def pop_world_sound(self) -> np.ndarray:
        prop = self._input.pop_sound()
        out  = np.clip(self._ambient_buffer + prop, 0.0, 1.0)
        self._ambient_buffer[:] = 0.0
        return out

    def add_sound(self, source_pos, sound_vector: np.ndarray) -> None:
        self._ambient_buffer = np.clip(
            self._ambient_buffer + np.asarray(sound_vector), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Vocalization
    # ------------------------------------------------------------------

    @property
    def last_agent_vocalization(self) -> np.ndarray:
        return self._last_agent_vocalization

    @last_agent_vocalization.setter
    def last_agent_vocalization(self, value: np.ndarray) -> None:
        self._last_agent_vocalization = value

    def receive_vocalization(self, vec: np.ndarray) -> None:
        self._last_agent_vocalization = vec

    # ------------------------------------------------------------------
    # Social stubs
    # ------------------------------------------------------------------

    def get_distance_to_human(self) -> float:
        return float('inf')

    def process_human_input(self, keys_pressed: dict) -> None:
        pass

    def get_render_string(self) -> str:
        return f"Game  tick={self.world_tick}"


# ---------------------------------------------------------------------------
# Default action set — WASD + mouse + common keys
# Covers most first/third-person games without configuration.
# ---------------------------------------------------------------------------

def _default_actions() -> dict[int, tuple]:
    return {
        0:  ('key',        'w'),          # move forward
        1:  ('key',        's'),          # move back
        2:  ('key',        'a'),          # strafe left
        3:  ('key',        'd'),          # strafe right
        4:  ('mouse',      'left'),       # attack / interact primary
        5:  ('mouse',      'right'),      # use / interact secondary
        6:  ('key',        'space'),      # jump
        7:  ('key',        'shift'),      # sneak / walk
        8:  ('mouse_move', (20, 0)),      # look right
        9:  ('mouse_move', (-20, 0)),     # look left
        10: ('mouse_move', (0, 15)),      # look down
        11: ('mouse_move', (0, -15)),     # look up
    }


# ---------------------------------------------------------------------------
# Probe helpers — pass as lambdas to homeostasis dict
# ---------------------------------------------------------------------------

def probe_red_bar(
    rgb: np.ndarray,
    x: int, y: int, w: int, h: int,
    threshold: int = 100,
) -> float:
    """Sample a rectangular region and return how 'red' it is [0,1].

    Useful for health bars in most action games (red = damage).
    pain = probe_red_bar(rgb, x=10, y=960, w=180, h=8)
    """
    if rgb is None:
        return 0.0
    region = rgb[y:y+h, x:x+w]
    if region.size == 0:
        return 0.0
    r = region[:, :, 0].astype(float)
    g = region[:, :, 1].astype(float)
    b = region[:, :, 2].astype(float)
    # Red dominance: r high, g and b low
    redness = np.mean(np.clip((r - g - b) / 255.0, 0.0, 1.0))
    return float(redness)


def probe_bar_fill(
    rgb: np.ndarray,
    x: int, y: int, w: int, h: int,
    color_channel: int = 0,
    invert: bool = False,
) -> float:
    """Measure how full a rectangular bar is by colour intensity [0,1].

    color_channel: 0=red, 1=green, 2=blue
    invert=True: empty bar = high signal (e.g. hunger bar empties)
    """
    if rgb is None:
        return 0.0
    region = rgb[y:y+h, x:x+w, color_channel].astype(float) / 255.0
    val = float(np.mean(region))
    return 1.0 - val if invert else val
