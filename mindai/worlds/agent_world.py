"""AgentWorld — multimodal sensory environment with curriculum learning.

Three training modes, mixed by curriculum ratio (see Brain construction):

  raw    — plain text corpus, no vision sync. Builds language statistics
           via STDP (child overhearing adult speech).
  paired — image/video + caption shown simultaneously. STDP binds the
           visual pattern to the token pattern (ostensive definition —
           adult pointing at object and naming it).
  qa     — Q/A pairs, no vision. Builds dialogue structure on top of
           grounded concepts.

Paired data layout (data/images/, data/video/):
    chair.png + chair.txt  ("I see a chair.")
    fire.png  + fire.txt   ("Fire is hot.")
Captions are optional; flat folders with matching basenames also work.
Each image is held for _PAIRED_HOLD_TICKS ticks while caption streams.
"""

from __future__ import annotations

import random
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN_NEURONS     = 2048
_IMAGE_EXTS       = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
_VIDEO_EXTS       = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
_PAIRED_HOLD_TICKS = 800   # ticks per image — enough for ~30 saccades (Yarbus 1967)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _build_token_patterns(vocab_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    patterns = np.zeros((vocab_size, TOKEN_NEURONS), dtype=np.float32)
    active = max(4, TOKEN_NEURONS // 50)
    for tid in range(vocab_size):
        idx = rng.choice(TOKEN_NEURONS, active, replace=False)
        patterns[tid, idx] = 1.0
    return patterns


def _nearest_token(motor: np.ndarray, patterns: np.ndarray) -> int:
    return int(np.argmax(patterns @ motor))


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_tokenizer(name: str = 'auto'):
    from mindai.worlds.tokenizers import get_tokenizer
    return get_tokenizer(name)


def _import_retina(vision_size: int):
    from mindai.worlds.minecraft.retina import FovealRetina
    return FovealRetina(vision_size=vision_size)


def _import_cochlea(audio_size: int):
    from mindai.environment.hearing_system import Cochlea
    return Cochlea(audio_size=audio_size)


# ---------------------------------------------------------------------------
# Media sources
# ---------------------------------------------------------------------------

# Human visual system: ~24 distinct frames/second, ~3-4 saccades/second.
_VIDEO_TARGET_FPS = 24.0


class _VideoFrameAudioSource:
    """Background thread: streams one video at 24 fps, extracts audio chunks.

    Reads video frames and audio simultaneously. Audio is fed into the
    shared Cochlea so the brain hears what it sees (Calvert 2001).
    Falls back to video-only if moviepy/av are not installed.
    """

    def __init__(self, path: Path, cochlea=None, audio_sr: int = 16000):
        self._path    = path
        self._cochlea = cochlea
        self._sr      = audio_sr
        self._frame: np.ndarray | None = None
        self._lock    = threading.Lock()
        self._done    = False
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def is_done(self) -> bool:
        return self._done

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def _run(self):
        try:
            import cv2
        except ImportError:
            self._done = True
            return

        # Try to extract audio first (non-blocking; failures are silent)
        audio_chunks: list[np.ndarray] = []
        self._try_extract_audio(audio_chunks)

        cap        = cv2.VideoCapture(str(self._path))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or _VIDEO_TARGET_FPS
        # skip factor: only decode every Nth frame to land near 24 fps
        skip       = max(1, round(native_fps / _VIDEO_TARGET_FPS))
        interval   = 1.0 / _VIDEO_TARGET_FPS

        frame_idx  = 0
        audio_idx  = 0
        t_next     = time.monotonic()

        while cap.isOpened():
            ok, fr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % skip != 0:
                continue

            with self._lock:
                self._frame = fr[:, :, ::-1].copy()   # BGR → RGB

            # Push matching audio chunk to cochlea
            if audio_chunks and self._cochlea is not None:
                chunk = audio_chunks[audio_idx % len(audio_chunks)]
                audio_idx += 1
                try:
                    result = self._cochlea.process(chunk)
                    with self._cochlea._lock:
                        self._cochlea._output[:] = result
                except Exception:
                    pass

            # Pace to 24 fps wall-clock (so downstream gets biological timing)
            t_next += interval
            sleep   = t_next - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

        cap.release()
        self._done = True

    def _try_extract_audio(self, out: list) -> None:
        """Fill `out` with audio chunks (one per video frame at 24fps)."""
        chunk_samples = int(self._sr / _VIDEO_TARGET_FPS)
        try:
            import av
            container = av.open(str(self._path))
            audio_stream = next(
                (s for s in container.streams if s.type == 'audio'), None)
            if audio_stream is None:
                return
            samples: list[np.ndarray] = []
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    arr = frame.to_ndarray()
                    if arr.ndim > 1:
                        arr = arr.mean(axis=0)
                    samples.append(arr.astype(np.float32))
            if samples:
                audio = np.concatenate(samples)
                for i in range(0, len(audio), chunk_samples):
                    blk = audio[i:i + chunk_samples]
                    if len(blk) < chunk_samples:
                        blk = np.pad(blk, (0, chunk_samples - len(blk)))
                    out.append(blk)
            container.close()
            return
        except Exception:
            pass

        try:
            import moviepy.editor as mp
            clip  = mp.VideoFileClip(str(self._path))
            if clip.audio is None:
                clip.close()
                return
            audio = clip.audio.to_soundarray(fps=self._sr)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            clip.close()
            for i in range(0, len(audio), chunk_samples):
                blk = audio[i:i + chunk_samples]
                if len(blk) < chunk_samples:
                    blk = np.pad(blk, (0, chunk_samples - len(blk)))
                out.append(blk)
        except Exception:
            pass


class _PairedSource:
    """Images and videos with optional captions for ostensive definition training.

    Images: held for _PAIRED_HOLD_TICKS ticks (static; STDP binds caption).
    Videos: streamed at 24 fps via _VideoFrameAudioSource (dynamic; STDP binds
            audio + caption + moving image simultaneously — Calvert 2001).
    """

    def __init__(
        self,
        images_dir: str | None = None,
        video_dir:  str | None = None,
        cochlea=None,
    ):
        self._pairs: list[tuple[Path, str | None]] = []
        self._cochlea = cochlea

        for folder, exts in [(images_dir, _IMAGE_EXTS), (video_dir, _VIDEO_EXTS)]:
            if not folder:
                continue
            p = Path(folder)
            for f in sorted(p.rglob('*')):
                if f.suffix.lower() not in exts:
                    continue
                txt     = f.with_suffix('.txt')
                caption = txt.read_text(encoding='utf-8').strip() if txt.exists() else None
                self._pairs.append((f, caption))

        if not self._pairs:
            raise FileNotFoundError('No images or videos found')
        # Sort by extension class so callers know what they're getting,
        # but shuffle within each class so order is randomised.
        random.shuffle(self._pairs)
        with_caption = sum(1 for _, c in self._pairs if c)
        print(f'>>> AgentWorld visual: {len(self._pairs)} files '
              f'({with_caption} with caption, {len(self._pairs)-with_caption} silent)')

        self._idx           = 0
        self._hold_ticks    = 0
        self._current_img:  np.ndarray | None = None
        self._current_cap:  str | None        = None
        self._video_stream: _VideoFrameAudioSource | None = None
        self._lock          = threading.Lock()
        self._advance()

    def _advance(self):
        if self._idx >= len(self._pairs):
            random.shuffle(self._pairs)
            self._idx = 0

        path, caption = self._pairs[self._idx]
        self._idx += 1

        if path.suffix.lower() in _VIDEO_EXTS:
            self._video_stream = _VideoFrameAudioSource(
                path, cochlea=self._cochlea)
            with self._lock:
                self._current_img = None   # will be filled by stream
                self._current_cap = caption
            self._hold_ticks = 0           # video advances itself; no hold
        else:
            self._video_stream = None
            try:
                import cv2
                img = cv2.imread(str(path))
                if img is not None:
                    with self._lock:
                        self._current_img = img[:, :, ::-1].copy()
                        self._current_cap = caption
            except ImportError:
                pass
            self._hold_ticks = _PAIRED_HOLD_TICKS

    @property
    def is_active(self) -> bool:
        """True while a paired item is currently being shown.

        Used by AgentWorld to lock curriculum mode and silence the token
        channel — so corpus.txt cannot leak into the brain's token-input
        while it is watching a captionless video or holding an image.
        """
        if self._video_stream is not None:
            return not self._video_stream.is_done
        return self._hold_ticks > 0

    def tick(self, tokens_remaining: int = 0) -> tuple[np.ndarray | None, str | None]:
        """Call each tick. Returns (frame, caption_or_None).

        For videos: caption returned once per video start; frame updated each tick.
        For images: held for _PAIRED_HOLD_TICKS; caption returned on first tick.
        """
        # Video mode — advance stream; move to next item when video ends
        if self._video_stream is not None:
            frame = self._video_stream.latest()
            if frame is not None:
                with self._lock:
                    self._current_img = frame
            cap = self._current_cap   # return caption once (first tick)
            self._current_cap = None  # clear so it doesn't repeat
            if self._video_stream.is_done and tokens_remaining == 0:
                self._advance()
            return frame, cap

        # Image mode
        self._hold_ticks -= 1
        cap = self._current_cap
        if self._hold_ticks <= 0 and tokens_remaining == 0:
            self._current_cap = None
            self._advance()
            with self._lock:
                return self._current_img, self._current_cap
        with self._lock:
            return self._current_img, cap if self._hold_ticks == _PAIRED_HOLD_TICKS - 1 else None


_AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


class _AudioFileSource:
    """Streams all audio files from a file or folder through the biological Cochlea."""

    def __init__(self, path: str, audio_size: int = 770, sr: int = 16000):
        from mindai.environment.hearing_system import Cochlea
        self._cochlea = Cochlea(audio_size=audio_size, sample_rate=sr)
        self._lock    = threading.Lock()
        self._out     = np.zeros(audio_size, dtype=np.float32)

        p = Path(path)
        if p.is_dir():
            self._files = sorted(f for f in p.rglob('*') if f.suffix.lower() in _AUDIO_EXTS)
        else:
            self._files = [p]

        if not self._files:
            print(f'>>> No audio files found in {path}')
            return

        print(f'>>> AgentWorld audio: {len(self._files)} file(s) in {path}')
        threading.Thread(target=self._loop, args=(sr,), daemon=True).start()

    def _loop(self, sr: int):
        chunk = self._cochlea.chunk_size
        try:
            import librosa
            loader = lambda p: librosa.load(str(p), sr=sr, mono=True)[0]
        except ImportError:
            try:
                import soundfile as sf
                loader = lambda p: sf.read(str(p), always_2d=False)[0]
            except ImportError:
                print('>>> pip install librosa  (audio disabled)')
                return

        idx = 0
        while True:
            f = self._files[idx % len(self._files)]
            idx += 1
            if idx >= len(self._files):
                random.shuffle(self._files)
                idx = 0
            try:
                y = loader(f)
                for i in range(0, len(y), chunk):
                    block = y[i:i + chunk]
                    if len(block) < chunk:
                        block = np.pad(block, (0, chunk - len(block)))
                    result = self._cochlea.process(block)
                    with self._lock:
                        self._out[:] = result
                    time.sleep(chunk / sr)
            except Exception as e:
                print(f'>>> audio error ({f.name}): {e}')
                time.sleep(1.0)

    def get(self) -> np.ndarray:
        with self._lock:
            return self._out.copy()


# ---------------------------------------------------------------------------
# AgentWorld
# ---------------------------------------------------------------------------

class AgentWorld:
    """Multimodal world with curriculum: raw text / paired / Q&A.

    Parameters
    ----------
    text_corpus : str | None
    images_dir  : str | None   Folder with images (+ optional .txt captions).
    video_dir   : str | None   Folder with videos (+ optional .txt captions).
    qa_corpus   : str | None
    audio_source: 'mic' | str | None   File or folder of audio files.
    vision_size : int          Must be divisible by 5.
    audio_size  : int
    interactive : bool
    curriculum  : tuple[float, float, float]
        (raw_weight, paired_weight, qa_weight). Default: (0.70, 0.25, 0.05).
    """

    def __init__(
        self,
        text_corpus:   str | None              = None,
        images_dir:    str | None              = None,
        video_dir:     str | None              = None,
        qa_corpus:     str | None              = None,
        audio_source:  str | None              = None,
        vision_size:   int                     = 2880,
        audio_size:    int                     = 770,
        interactive:   bool                    = True,
        max_context:   int                     = 64,
        curriculum:    tuple[float,float,float]= (0.70, 0.25, 0.05),
        tokenizer:     str                     = 'auto',
        text_stream                            = None,
        phase_ticks:   dict[str, int] | None   = None,
    ):
        self._token_n     = TOKEN_NEURONS
        self._v_size      = vision_size
        self._a_size      = audio_size
        self._max_context = max_context
        self._interactive = interactive

        # Curriculum weights → cumulative thresholds for random draw
        r, p, q = curriculum
        total = r + p + q
        self._thresh_raw    = r / total
        self._thresh_paired = (r + p) / total

        # ---- tokenizer & patterns ----------------------------------------
        self._tokenizer = _import_tokenizer(tokenizer)
        self._patterns  = _build_token_patterns(self._tokenizer.vocab_size)

        # ---- raw corpus — paragraph mode (Chomsky 1965: continuity of thought)
        # Split on blank lines so each "line" is a full paragraph, not a sentence.
        # STDP sees a continuous token stream within a paragraph before switching.
        self._raw_lines: list[str] = []
        if text_corpus and Path(text_corpus).exists():
            with open(text_corpus, encoding='utf-8', errors='ignore') as f:
                text = f.read()
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
            if paragraphs:
                self._raw_lines = paragraphs
            else:
                # File has no blank-line separators — fall back to line-by-line
                self._raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
            print(f'>>> AgentWorld raw: {len(self._raw_lines)} paragraphs')
        else:
            from mindai.worlds.text_world import _SEED_CORPUS
            self._raw_lines = list(_SEED_CORPUS)
            print('>>> AgentWorld raw: seed corpus')
        self._raw_idx = 0
        self._text_stream = iter(text_stream) if text_stream is not None else None

        # ---- Q&A corpus --------------------------------------------------
        self._qa_pairs: list[tuple[str, str]] = []
        if qa_corpus and Path(qa_corpus).exists():
            lines = [l.strip() for l in
                     Path(qa_corpus).read_text(encoding='utf-8').splitlines()
                     if l.strip()]
            self._qa_pairs = [(lines[i], lines[i+1])
                              for i in range(0, len(lines)-1, 2)]
            print(f'>>> AgentWorld Q&A: {len(self._qa_pairs)} pairs')
        self._qa_idx = 0

        # ---- audio (must come before paired so cochlea is ready) -----------
        self._cochlea    = None
        self._audio_file = None
        self._prev_audio = np.zeros(audio_size, dtype=np.float32)
        if audio_source == 'mic':
            self._cochlea = _import_cochlea(audio_size)
            self._cochlea.start_mic()
        elif audio_source:
            self._audio_file = _AudioFileSource(audio_source, audio_size=audio_size)

        # If video data is present we need a shared cochlea for video-frame
        # audio to land somewhere — _VideoFrameAudioSource pushes into it.
        # Mic is NOT started here; the cochlea is purely a receiver.
        _video_dir_present = video_dir and Path(video_dir).is_dir()
        if _video_dir_present and self._cochlea is None:
            self._cochlea = _import_cochlea(audio_size)

        # ---- paired (ostensive) — images and videos as SEPARATE sources -----
        # Kept apart so the curriculum can introduce them in phases:
        # text → images (+text) → video (+images+text).
        # Mixing video and text simultaneously without a caption was teaching
        # STDP to bind unrelated corpus tokens onto POV-video frames.
        self._paired_images: _PairedSource | None = None
        self._paired_videos: _PairedSource | None = None
        has_images = images_dir and Path(images_dir).is_dir()
        has_video  = video_dir  and Path(video_dir).is_dir()
        if has_images:
            try:
                self._paired_images = _PairedSource(
                    images_dir=images_dir, video_dir=None,
                    cochlea=self._cochlea)
            except FileNotFoundError:
                self._paired_images = None
        if has_video:
            try:
                self._paired_videos = _PairedSource(
                    images_dir=None, video_dir=video_dir,
                    cochlea=self._cochlea)
            except FileNotFoundError:
                self._paired_videos = None

        # ---- curriculum phase machine ------------------------------------
        # Strict sequential phases — ONE source active at a time, no mixing.
        # Order: c4 → text → images → video → audio → qa.
        # Each phase silences inputs that don't belong to it (token channel,
        # audio channel) so STDP never builds anti-bindings between modalities
        # that just happen to co-occur in wall-clock time.
        # Phases without data are skipped automatically.
        # Defaults assume 10 Hz; tweak via phase_ticks={'name': start_tick}.
        default_pt = {
            'c4':     0,
            'text':   30_000,
            'images': 60_000,
            'video':  120_000,
            'audio':  200_000,
            'qa':     250_000,
        }
        self._phase_starts = dict(default_pt)
        if phase_ticks:
            self._phase_starts.update(phase_ticks)
        # Canonical ordering — used for both promotion and skipping
        self._phase_order = ('c4', 'text', 'images', 'video', 'audio', 'qa')
        # Initial phase: first available
        self._phase: str = self._first_available_phase()
        self._phase_announced: set[str] = set()

        # ---- vision (retina) ---------------------------------------------
        self._retina = None
        has_vision   = (self._paired_images is not None
                        or self._paired_videos is not None)
        if has_vision:
            if vision_size % 5 != 0:
                raise ValueError('vision_size must be divisible by 5')
            self._retina = _import_retina(vision_size)
            self._retina._capture_rgb = self._get_current_frame
            print(f'>>> AgentWorld retina: {vision_size} neurons')

        # Eye fixation — set by SuperiorColliculus via receive_gaze()
        self._eye_fixation: tuple[float, float] = (0.0, 0.0)

        # ---- injected media (interactive chat) ---------------------------
        self._injected_frame: np.ndarray | None           = None
        self._injected_video: _VideoFrameAudioSource | None = None
        self._inject_hold:    int                          = 0   # -1 = infinite

        # ---- async stream state ------------------------------------------
        # When True, output tokens are printed as they are generated,
        # simulating the brain "thinking aloud" while looking at an image.
        self._streaming: bool      = False
        self._stream_printed: int  = 0      # chars already sent to stdout

        # ---- token stream state ------------------------------------------
        self._input_queue:   list[int] = []
        self._context:       deque[int] = deque(maxlen=max_context)
        self._current_token: int       = 0
        self._next_token:    int       = 0
        self._motor_pattern  = np.zeros(TOKEN_NEURONS, dtype=np.float32)
        self._output_ids:    list[int] = []
        self._output_text:   str       = ''
        self._pending_response_at: int = -1
        self._tick_counter: list[int]  = [0]

        # Current curriculum mode for this tick
        self._mode: str = 'raw'
        self._paired_frame: np.ndarray | None = None

        self._load_next_token()

        # ---- brain.py stubs ----------------------------------------------
        self._sound_queue: list[np.ndarray] = []
        self.last_agent_vocalization = np.zeros(32, dtype=np.float32)
        self.isolation_ticks = 0
        self.agent_pos  = [0, 0]
        self.human_pos  = [0, 0]
        self.world_tick = 0

        if interactive:
            self._start_input_thread()
            self._start_stream_thread()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    @property
    def sensory_layout(self) -> dict[str, int]:
        layout: dict[str, int] = {}
        if self._retina:
            layout['vision'] = self._v_size
        if self._cochlea or self._audio_file:
            layout['audio']  = self._a_size
        layout['token'] = self._token_n * 2
        return layout

    @property
    def motor_layout(self) -> dict[str, int]:
        return {'motor': self._token_n}

    @property
    def tokenizer(self):
        return self._tokenizer

    # -----------------------------------------------------------------------
    # Frame routing
    # -----------------------------------------------------------------------

    def _get_current_frame(self) -> np.ndarray:
        # Injected media (interactive chat) takes priority over training curriculum
        if self._injected_video is not None:
            frame = self._injected_video.latest()
            if frame is not None:
                self._injected_frame = frame
            if self._injected_video.is_done:
                self._injected_video = None
                if self._inject_hold != -1:
                    self._injected_frame = None

        if self._injected_frame is not None:
            if self._inject_hold > 0:
                self._inject_hold -= 1
            elif self._inject_hold == 0:
                self._injected_frame = None
            if self._injected_frame is not None:
                return self._injected_frame

        # Vision is only emitted during paired phases (images/video).
        if (self._phase in ('images', 'video')
                and self._paired_frame is not None):
            return self._paired_frame
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # -----------------------------------------------------------------------
    # World interface
    # -----------------------------------------------------------------------

    def get_homeostatic_signals(self) -> dict[str, float]:
        return {'pain': 0.0, 'hunger': 0.0, 'thirst': 0.0}

    # -----------------------------------------------------------------------
    # Curriculum phase machine
    # -----------------------------------------------------------------------

    def _phase_has_data(self, phase: str) -> bool:
        """True if the data source for `phase` is actually configured."""
        if phase == 'c4':     return self._text_stream is not None
        if phase == 'text':   return bool(self._raw_lines)
        if phase == 'images': return self._paired_images is not None
        if phase == 'video':  return self._paired_videos is not None
        if phase == 'audio':  return (self._cochlea is not None
                                      or self._audio_file is not None)
        if phase == 'qa':     return bool(self._qa_pairs)
        return False

    def _first_available_phase(self) -> str:
        for p in self._phase_order:
            if self._phase_has_data(p):
                return p
        return 'text'   # safe fallback

    def _maybe_advance_phase(self) -> None:
        """Promote phase by tick count along the canonical order.

        Phases with no configured data source are skipped.
        Each phase crossing is announced once in the log.
        """
        t = self._tick_counter[0]
        cur_idx = self._phase_order.index(self._phase)
        # Walk forward from current phase, advancing as long as the next
        # phase's start tick has been reached.
        for nxt in self._phase_order[cur_idx + 1:]:
            if t < self._phase_starts.get(nxt, 1 << 60):
                break
            if self._phase_has_data(nxt):
                self._phase = nxt
        if self._phase not in self._phase_announced:
            print(f'\n>>> [CURRICULUM] фаза → {self._phase} '
                  f'на тике {t:,}\n')
            self._phase_announced.add(self._phase)

    def _pick_phase_source(self) -> None:
        """Strict phase selection — exactly one active source per phase.

        c4     — streaming text dataset (allenai/c4 etc.) via _text_stream
        text   — corpus.txt only
        images — paired_images only (captions feed token channel)
        video  — paired_videos only (their own audio feeds audio channel)
        audio  — audio stream only (no vision, no tokens)
        qa     — qa.txt only

        _load_next_token consults self._phase to decide which text source
        to pull from (c4 stream vs corpus paragraphs). Other channels are
        gated in get_sensory_retina.
        """
        if self._phase in ('c4', 'text'):
            self._mode = self._phase           # 'c4' or 'text'
            return

        if self._phase == 'images':
            if self._paired_images is not None:
                self._activate_paired(self._paired_images, 'paired_image')
            else:
                self._mode = 'silent'
            return

        if self._phase == 'video':
            if self._paired_videos is not None:
                self._activate_paired(self._paired_videos, 'paired_video')
            else:
                self._mode = 'silent'
            return

        if self._phase == 'audio':
            # No vision, no tokens — just listen.
            self._mode = 'audio'
            return

        if self._phase == 'qa':
            if self._qa_pairs:
                q, a = self._qa_pairs[self._qa_idx % len(self._qa_pairs)]
                self._qa_idx += 1
                self._input_queue.extend(self._tokenizer.encode(q + ' ' + a))
                self._mode = 'qa'
            else:
                self._mode = 'silent'
            return

        self._mode = 'silent'

    def _activate_paired(self, source: '_PairedSource', mode: str) -> None:
        frame, caption = source.tick(tokens_remaining=0)
        self._paired_frame = frame
        if caption:
            self._input_queue.extend(self._tokenizer.encode(caption))
        self._mode = mode

    def get_sensory_retina(self, num_neurons: int) -> np.ndarray:
        # Advance curriculum phase if its tick threshold has been crossed
        self._maybe_advance_phase()

        # If a paired item is already on screen, keep it on screen and just
        # advance its frame — do NOT redraw curriculum, do NOT pull corpus
        # tokens. This is the fix for the "corpus.txt leaking into video"
        # anti-binding bug.
        media_active = False
        if self._paired_images is not None and self._paired_images.is_active:
            frame, caption = self._paired_images.tick(
                tokens_remaining=len(self._input_queue))
            self._paired_frame = frame
            if caption:
                self._input_queue.extend(self._tokenizer.encode(caption))
            self._mode = 'paired_image'
            media_active = True
        elif self._paired_videos is not None and self._paired_videos.is_active:
            frame, caption = self._paired_videos.tick(
                tokens_remaining=len(self._input_queue))
            self._paired_frame = frame
            if caption:
                self._input_queue.extend(self._tokenizer.encode(caption))
            self._mode = 'paired_video'
            media_active = True

        # Otherwise pick a fresh source for this tick — but only if the
        # token queue is empty (don't interrupt a sentence mid-stream).
        if not media_active and not self._input_queue:
            self._pick_phase_source()

        parts: list[np.ndarray] = []

        # Vision — fixation comes from brain's SC motor output, not a script
        if self._retina is not None:
            self._retina.fixation = self._eye_fixation
            try:
                parts.append(self._retina.get_visual_array())
            except Exception:
                parts.append(np.zeros(self._v_size, dtype=np.float32))

        # Audio — strict phase gating. The audio channel is only fed in
        # the 'video' and 'audio' phases. During earlier phases (c4, text,
        # images) any background source (mic, _AudioFileSource thread) is
        # ignored so STDP cannot bind unrelated audio onto whatever else
        # is active. The channel still exists in the sensory layout so we
        # emit zeros to keep shapes consistent.
        if self._cochlea is not None or self._audio_file is not None:
            zero_spec = np.zeros(self._a_size, dtype=np.float32)
            if self._phase == 'video' and self._cochlea is not None:
                spec = self._cochlea.get_auditory_nerve_signal()
            elif self._phase == 'audio':
                if self._audio_file is not None:
                    spec = self._audio_file.get()
                elif self._cochlea is not None:
                    spec = self._cochlea.get_auditory_nerve_signal()
                else:
                    spec = zero_spec
            else:
                spec = zero_spec
            parts.append(np.concatenate([spec, self._prev_audio]).astype(np.float32))
            self._prev_audio = spec.copy()

        # Token
        cur_pat = self._patterns[self._current_token % len(self._patterns)]
        ctx_pat = np.zeros(self._token_n, dtype=np.float32)
        if self._context:
            ctx_pat = self._patterns[self._context[-1] % len(self._patterns)] * 0.7
        parts.append(np.concatenate([cur_pat, ctx_pat]))

        full = np.concatenate(parts).astype(np.float32) if parts else np.array([], dtype=np.float32)
        out  = np.zeros(num_neurons, dtype=np.float32)
        n    = min(len(full), num_neurons)
        out[:n] = full[:n]
        return out

    def receive_motor_pattern(self, motor_signals: np.ndarray) -> None:
        n = min(len(motor_signals), self._token_n)
        self._motor_pattern[:n] = motor_signals[:n]
        if n < self._token_n:
            self._motor_pattern[n:] = 0.0

    def receive_gaze(self, fx: float, fy: float) -> None:
        """Called by SuperiorColliculus with the new fixation point.

        fx, fy are in normalised image coords [-1, 1].
        The SC computed these from the brain's internal state — desire, surprise,
        threat, dopamine — not from a scripted rule.
        """
        self._eye_fixation = (float(fx), float(fy))

    def execute_action(self, motor_idx: int) -> dict:
        if float(np.linalg.norm(self._motor_pattern)) > 0.01:
            predicted = _nearest_token(self._motor_pattern, self._patterns)
        else:
            predicted = motor_idx % self._tokenizer.vocab_size

        self._context.append(self._current_token)

        correct = (predicted == self._next_token)
        self._output_ids.append(predicted)
        if len(self._output_ids) > 200:
            self._output_ids = self._output_ids[-200:]
        self._output_text = self._tokenizer.decode(self._output_ids[-80:])

        self._load_next_token()
        t = self._tick_counter[0] + 1
        self._tick_counter[0] = t
        self.world_tick += 1

        if self._pending_response_at > 0 and t >= self._pending_response_at:
            print(f'\nBrain [{t:,}]: {self._output_text[-200:]}')
            self._pending_response_at = -1

        return {
            'energy': 0.3 if correct else 0.0,
            'water':  0.0,
            'stress': 0.0 if correct else 0.1,
        }

    def is_alive(self) -> bool:
        return getattr(self, '_alive', True)

    def get_current_output(self) -> str:
        return self._output_text

    def inject_prompt(self, text: str) -> None:
        ids = self._tokenizer.encode(text)
        self._input_queue.extend(ids)
        self._pending_response_at = self._tick_counter[0] + len(ids) + 100
        print(f'>>> Промпт: {len(ids)} токенов')

    # -----------------------------------------------------------------------
    # Stubs
    # -----------------------------------------------------------------------

    def process_human_input(self, keys: dict) -> None:
        text = keys.get('text_input', '')
        if text:
            self.inject_prompt(text)

    def get_distance_to_human(self) -> float:
        return float('inf')

    def pop_world_sound(self) -> np.ndarray:
        if self._sound_queue:
            return self._sound_queue.pop(0)
        return np.zeros(32, dtype=np.float32)

    def add_sound(self, pos, sound: np.ndarray) -> None:
        pass

    def receive_vocalization(self, vocal: np.ndarray) -> None:
        self.last_agent_vocalization = vocal

    # -----------------------------------------------------------------------
    # Interactive image injection
    # -----------------------------------------------------------------------

    def inject_image(
        self,
        source,
        hold_ticks: int | None = None,
    ) -> None:
        """Inject a static image or video into the visual field.

        source : str | Path | np.ndarray | None
            Image/video file path, RGB numpy array (H,W,3), or None to clear.
        hold_ticks : int | None
            Ticks to hold. None = hold indefinitely.
        """
        if source is None:
            self._injected_frame   = None
            self._injected_video   = None
            self._inject_hold      = 0
            self._streaming        = False
            return

        path = Path(source) if isinstance(source, (str, Path)) else None

        # --- Video file ---
        if path is not None and path.suffix.lower() in _VIDEO_EXTS:
            self._injected_video = _VideoFrameAudioSource(
                path, cochlea=self._cochlea)
            self._injected_frame = None
            self._inject_hold    = hold_ticks if hold_ticks is not None else -1
            self._streaming      = True
            self._stream_printed = len(self._output_text)
            print(f'\n>>> [Видео] {path.name} — смотрю...\n')
            return

        # --- Image file or numpy array ---
        if path is not None:
            try:
                import cv2
                img = cv2.imread(str(path))
                if img is None:
                    print(f'>>> inject_image: не удалось загрузить {path}')
                    return
                frame = img[:, :, ::-1].copy()
            except ImportError:
                print('>>> pip install opencv-python')
                return
        else:
            frame = np.asarray(source)

        self._injected_frame = frame
        self._injected_video = None
        self._inject_hold    = hold_ticks if hold_ticks is not None else -1
        self._streaming      = True
        self._stream_printed = len(self._output_text)
        print(f'\n>>> [Изображение {frame.shape[1]}×{frame.shape[0]}] '
              f'смотрю...\n')

    def inject_audio(self, source) -> None:
        """Inject an audio file into the auditory channel during chat.

        source : str | Path
            Audio file path (.wav, .mp3, .flac, ...).
        """
        if self._cochlea is None:
            print('>>> inject_audio: cochlea не инициализирована '
                  '(запусти с audio_source)')
            return
        path = Path(source)
        if not path.exists():
            print(f'>>> inject_audio: файл не найден {path}')
            return
        # Reuse _AudioFileSource for one-shot playback
        _AudioFileSource(str(path), audio_size=self._a_size)
        print(f'>>> [Аудио] {path.name}')

    def _start_stream_thread(self) -> None:
        """Background thread: print new output tokens as they are generated.

        Only active while self._streaming is True (image is being examined).
        Simulates asynchronous speech while the brain explores the image.
        """
        def _runner():
            while True:
                time.sleep(0.3)
                if not self._streaming:
                    continue
                text = self._output_text
                if len(text) > self._stream_printed:
                    chunk = text[self._stream_printed:]
                    self._stream_printed = len(text)
                    print(chunk, end='', flush=True)
                # Stop streaming when image is gone and token queue is empty
                if (self._injected_frame is None
                        and not self._input_queue):
                    if self._streaming:
                        print()   # newline after last token
                    self._streaming = False
        threading.Thread(target=_runner, daemon=True).start()

    # -----------------------------------------------------------------------
    # Token loading
    # -----------------------------------------------------------------------

    def _load_next_token(self) -> None:
        if self._input_queue:
            self._current_token = self._next_token
            self._next_token    = self._input_queue.pop(0)
            return
        # Silence token channel whenever paired media is on screen with no
        # queued caption tokens, or during interactive injection / pending
        # response. This blocks the curriculum-mix anti-binding bug:
        # corpus.txt tokens must NEVER fire during a captionless video.
        paired_active = (
            (self._paired_images is not None and self._paired_images.is_active)
            or (self._paired_videos is not None and self._paired_videos.is_active)
        )
        if self._streaming or self._pending_response_at > 0 or paired_active:
            self._current_token = self._next_token
            self._next_token    = 0
            return
        # Outside the two text-streaming phases the token channel is silent.
        # paired phases handle their own caption tokens through _input_queue;
        # qa phase queues from qa_pairs; audio/silent phases emit nothing.
        if self._phase not in ('c4', 'text'):
            self._current_token = self._next_token
            self._next_token    = 0
            return

        # Continue reading current paragraph if tokens remain
        if (hasattr(self, '_corpus_tokens')
                and self._corpus_token_idx < len(self._corpus_tokens)):
            self._current_token    = self._next_token
            self._next_token       = self._corpus_tokens[self._corpus_token_idx]
            self._corpus_token_idx += 1
            return
        # Load next paragraph from the source dictated by current phase
        while True:
            line = None
            if self._phase == 'c4' and self._text_stream is not None:
                try:
                    line = next(self._text_stream)
                except StopIteration:
                    self._text_stream = None
                except Exception as e:
                    print(f'>>> text_stream error: {e}; phase c4 exhausted')
                    self._text_stream = None
            elif self._phase == 'text' and self._raw_lines:
                if self._raw_idx >= len(self._raw_lines):
                    self._raw_idx = 0
                    random.shuffle(self._raw_lines)
                line = self._raw_lines[self._raw_idx]
                self._raw_idx += 1
            if line is None:
                # No data this tick — emit silence
                self._current_token = self._next_token
                self._next_token    = 0
                return
            ids = self._tokenizer.encode(line)
            if ids:
                self._corpus_tokens    = ids
                self._corpus_token_idx = 0
                break
        self._current_token    = self._next_token
        self._next_token       = self._corpus_tokens[self._corpus_token_idx]
        self._corpus_token_idx += 1

    # -----------------------------------------------------------------------
    # Interactive stdin
    # -----------------------------------------------------------------------

    def _start_input_thread(self) -> None:
        def _reader():
            print('\n>>> AgentWorld готов. Введите текст и нажмите Enter.')
            print('>>> Чтобы показать изображение: введите путь к файлу')
            print('>>> Комбо: "path/to/img.jpg что рядом с деревом?"')
            print('>>> (Ctrl+C для выхода)\n')
            while True:
                try:
                    line = input('You: ').strip()
                    if not line:
                        continue

                    # Detect "file_path [optional text]" syntax
                    parts     = line.split(None, 1)
                    candidate = Path(parts[0])
                    if candidate.exists():
                        ext = candidate.suffix.lower()
                        if ext in _IMAGE_EXTS | _VIDEO_EXTS:
                            self.inject_image(candidate)
                            if len(parts) > 1 and parts[1].strip():
                                self.inject_prompt(parts[1])
                        elif ext in _AUDIO_EXTS:
                            self.inject_audio(candidate)
                            if len(parts) > 1 and parts[1].strip():
                                self.inject_prompt(parts[1])
                        else:
                            self.inject_prompt(line)
                    else:
                        self.inject_prompt(line)

                except (EOFError, KeyboardInterrupt):
                    break
        threading.Thread(target=_reader, daemon=True).start()
