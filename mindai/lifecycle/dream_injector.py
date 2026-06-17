"""DreamInjector — injects pre-recorded video into REM dream activity.

Biological basis
----------------
During REM sleep the prefrontal cortex is deactivated (Hobson 2009, Nir &
Tononi 2010). The brain cannot distinguish injected visual patterns from
internally generated ones — it processes them identically, including:
- Dopamine release when the dream shows eating/reward (VTA activation)
- Substance-P sensitisation when the dream shows pain
- STDP consolidation of vision→motor paths (Stickgold 2001)

This is the computational equivalent of Targeted Memory Reactivation (TMR,
Rudoy 2009; Rasch 2007) extended to full visual sequences.

The injected frames pass through the SAME FovealRetina sampling pipeline as
waking vision — the brain cannot tell them apart.

Usage
-----
    from mindai.lifecycle.dream_injector import DreamInjector
    from mindai.environment.retina import FovealRetina

    retina   = FovealRetina(vision_size=2880)
    injector = DreamInjector('videos/chop_wood.mp4', retina, fps=10)

    sleep_cycle.set_dream_injector(injector)

Video format
------------
Any codec readable by OpenCV (MP4, AVI, MOV …).  The injector loops the
video indefinitely so short clips are fine.  Recommend 10–30 fps clips of
10–60 seconds each; they will be looped across all REM periods.
"""

from __future__ import annotations

import numpy as np


class DreamInjector:
    """Preprocesses a video file into retina-format frames for REM injection.

    Parameters
    ----------
    video_path:
        Path to video file (MP4, AVI, etc.).
    retina:
        FovealRetina instance — used to sample each frame through the same
        non-uniform foveal grid as waking vision.
    fps:
        Target playback speed during dreams (frames per dream tick).
        Default 1 = one new video frame per REM tick.
    blend_weight:
        How strongly to blend injected content into dream activity [0–1].
        1.0 = fully replace PGO noise with video; 0.5 = equal mix.
        Recommended: 0.7 (video dominant but retains some spontaneous activity).
    """

    def __init__(
        self,
        video_path:   str,
        retina,
        fps:          float = 1.0,
        blend_weight: float = 0.7,
    ) -> None:
        self.blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
        self._fps         = max(0.1, fps)
        self._tick        = 0.0
        self._frames: list[np.ndarray] = []
        self._frame_idx   = 0

        self._load_frames(video_path, retina)

    # ------------------------------------------------------------------

    def _load_frames(self, video_path: str, retina) -> None:
        try:
            import cv2
        except ImportError:
            raise ImportError("pip install opencv-python")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Sample every Nth frame to match target dream fps
        step = max(1, int(round(video_fps / self._fps)))

        frame_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                # Convert BGR → RGB float32
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                # Pass through FovealRetina sampling (same pipeline as waking vision)
                vec = retina.preprocess_image(rgb)
                self._frames.append(vec)
            frame_idx += 1

        cap.release()

        if not self._frames:
            raise ValueError(f"No frames extracted from {video_path}")

        print(f">>> DreamInjector: {len(self._frames)} dream frames loaded "
              f"from '{video_path}' (blend={self.blend_weight})")

    def next_frame(self) -> np.ndarray:
        """Return the next preprocessed vision vector, looping the video."""
        frame = self._frames[self._frame_idx % len(self._frames)]
        self._frame_idx += 1
        return frame

    def reset(self) -> None:
        """Reset playback to start (called at each new sleep cycle)."""
        self._frame_idx = 0
