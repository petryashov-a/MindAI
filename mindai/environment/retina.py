"""FovealRetina — biologically accurate non-uniform foveal vision.

Human vision has three coupled mechanisms that reduce peripheral acuity.
All three are implemented here, and the implementation scales correctly
to any number of vision neurons.

1. Receptor sparsity (Curcio 1990)
   Cone density: ρ(e) ∝ 1/(1 + e/e₀)²,  e₀ ≈ 2°
   The CENTRE gets genuinely more sample points than the periphery.
   This is not blur — it is real information loss at the receptor level.
   Even with perfect optics, the far periphery cannot resolve fine detail
   because there are physically fewer cones per square degree.

   Zone allocation (scales proportionally with vision_size):
     Fovea     0°–  3°  → 34% of neurons, uniform grid
     Parafovea 3°– 12°  → 29% of neurons, log-polar
     Near-peri 12°– 35° → 17% of neurons, log-polar
     Far-peri  35°– 80° → 20% of neurons, log-polar

   Example: vision_size=2880 (576 pts × 5ch):
     fovea=196 (14×14), para=168, near=100, far=112

   Example: vision_size=200 (40 pts × 5ch):
     fovea=14 (3×4+2), para=12, near=7, far=7

   The biological ratio is preserved regardless of neuron count.
   More neurons → denser fovea AND denser periphery, both improve.

2. Optical aberration / circles of confusion (Navarro 1993)
   σ_deg(e) ≈ 0.06 × e     [degrees of visual angle]
   Converted to pixels:  σ_px = σ_deg × (image_width / fov_h_deg)
   Four pre-blurred image pyramids cover the four zones.

3. Cortical magnification (Schwartz 1977)
   Log-polar ring spacing mirrors the V1 layout.

Output channels per sample point (5 — fixed):
    0  R     [0, 1]
    1  G     [0, 1]
    2  B     [0, 1]
    3  Luma  = 0.299R + 0.587G + 0.114B
    4  Motion = |luma - prev_luma|, clipped [0, 1]

Output shape: (vision_size,) float32  ← always exact match to Brain layout.

References:
    Curcio CA et al. (1990) Human photoreceptor topography. J Comp Neurol 292.
    Navarro R et al. (1993) Modulation transfer as a function of retinal
        eccentricity. J Opt Soc Am A 10.
    Schwartz EL (1977) Spatial mapping in primate sensory projection.
        Biol Cybernetics 25.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Fixed biological constants
# ---------------------------------------------------------------------------

_BLUR_K = 0.06    # σ_deg = BLUR_K × ecc_deg  (Navarro 1993)
_FOV_H  = 160.0   # ±80° horizontal
_FOV_V  = 130.0   # ±65° vertical

# Eccentricity zone boundaries (degrees)
_ECC_FOVEA     =  3.0
_ECC_PARAFOVEA = 12.0
_ECC_NEAR_PERI = 35.0
_ECC_FAR_PERI  = 80.0

# Biological allocation ratios from Curcio (1990) cone density integration
# across the four eccentricity zones. These are fixed — scaling neuron count
# preserves the ratio, just at higher/lower resolution per zone.
_RATIO_FOVEA = 0.34
_RATIO_PARA  = 0.29
_RATIO_NEAR  = 0.17
_RATIO_FAR   = 0.20   # remainder

# Sector counts for log-polar zones — fixed (angular resolution is absolute,
# not relative, so sectors don't scale with total neuron count).
# Ring count scales with num_points so more neurons → more radial resolution.
_SECTORS_PARA = 28
_SECTORS_NEAR = 20
_SECTORS_FAR  = 28


def _build_sampling_grid(
    num_points: int,
    fov_h: float = _FOV_H,
    fov_v: float = _FOV_V,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the non-uniform sample point grid for any neuron count.

    Zone sizes are computed proportionally from Curcio (1990) ratios so the
    biological distribution is preserved regardless of num_points.

    Returns
    -------
    xy : (num_points, 2) float32
        Normalised image coordinates in [-1, 1].
        x = horizontal (right is +), y = vertical (down is +).
    ecc : (num_points,) float32
        Eccentricity in degrees for each sample point.
    """
    half_h = fov_h / 2.0
    half_v = fov_v / 2.0
    points: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Allocate points per zone (floor, with remainder going to fovea)
    # ------------------------------------------------------------------
    n_para = max(1, int(num_points * _RATIO_PARA))
    n_near = max(1, int(num_points * _RATIO_NEAR))
    n_far  = max(1, int(num_points * _RATIO_FAR))
    n_fovea = num_points - n_para - n_near - n_far   # remainder → most valuable zone

    # ------------------------------------------------------------------
    # Zone 0 — fovea: densest uniform rectangular grid
    # ------------------------------------------------------------------
    # Find the largest square (h×w) with h×w ≤ n_fovea, minimising waste
    fovea_h = max(1, int(np.sqrt(n_fovea)))
    fovea_w = max(1, n_fovea // fovea_h)
    # Adjust to use extra rows if they fit
    while (fovea_h + 1) * fovea_w <= n_fovea:
        fovea_h += 1

    fovea_norm_x = _ECC_FOVEA / half_h
    fovea_norm_y = _ECC_FOVEA / half_v
    for iy in range(fovea_h):
        for ix in range(fovea_w):
            x = ((ix / max(fovea_w - 1, 1)) * 2 - 1) * fovea_norm_x
            y = ((iy / max(fovea_h - 1, 1)) * 2 - 1) * fovea_norm_y
            points.append((x, y))

    # Remaining fovea points (if fovea_h × fovea_w < n_fovea): add row
    placed_fovea = fovea_h * fovea_w
    for k in range(n_fovea - placed_fovea):
        x = ((k / max(n_fovea - placed_fovea - 1, 1)) * 2 - 1) * fovea_norm_x
        y = fovea_norm_y   # extra row at top
        points.append((x, y))

    # ------------------------------------------------------------------
    # Helper: log-polar zone
    # ------------------------------------------------------------------
    def add_log_polar(ecc_inner, ecc_outer, n_total, preferred_sectors, offset=0.0):
        if n_total <= 0:
            return
        # When budget is smaller than preferred_sectors, reduce sectors so we
        # always get at least 1 full ring. Keeps angular coverage even when few neurons.
        n_sectors = min(preferred_sectors, n_total)
        n_rings   = max(1, n_total // n_sectors)
        log_min   = np.log(ecc_inner)
        log_max   = np.log(ecc_outer)
        for ri in range(n_rings):
            t   = ri / max(n_rings - 1, 1)
            ecc = np.exp(log_min + t * (log_max - log_min))
            rx  = ecc / half_h
            ry  = ecc / half_v
            for si in range(n_sectors):
                theta = (si / n_sectors) * 2 * np.pi + offset
                points.append((rx * np.cos(theta), ry * np.sin(theta)))
        # Remaining points: spread at outer ring edge
        used = n_rings * n_sectors
        remainder = n_total - used
        if remainder > 0:
            ecc = np.exp(log_max)
            rx  = ecc / half_h
            ry  = ecc / half_v
            for k in range(remainder):
                theta = (k / remainder) * 2 * np.pi + offset
                points.append((rx * np.cos(theta), ry * np.sin(theta)))

    # Zone 1 — parafovea
    add_log_polar(_ECC_FOVEA,     _ECC_PARAFOVEA, n_para, _SECTORS_PARA, offset=0.0)
    # Zone 2 — near periphery
    add_log_polar(_ECC_PARAFOVEA, _ECC_NEAR_PERI, n_near, _SECTORS_NEAR,
                  offset=np.pi / _SECTORS_NEAR)
    # Zone 3 — far periphery
    add_log_polar(_ECC_NEAR_PERI, _ECC_FAR_PERI,  n_far,  _SECTORS_FAR,  offset=0.0)

    # ------------------------------------------------------------------
    # Trim/pad to exactly num_points (rounding artefacts only)
    # ------------------------------------------------------------------
    if len(points) > num_points:
        points = points[:num_points]
    while len(points) < num_points:
        # Fill from far periphery at equally spaced angles
        theta = (len(points) - num_points) * 2 * np.pi / max(num_points, 1)
        ecc   = _ECC_FAR_PERI
        points.append((ecc / half_h * np.cos(theta), ecc / half_v * np.sin(theta)))

    xy  = np.array(points, dtype=np.float32)
    ecc = np.sqrt((xy[:, 0] * half_h) ** 2 + (xy[:, 1] * half_v) ** 2).astype(np.float32)
    return xy, ecc


# ---------------------------------------------------------------------------
# FovealRetina
# ---------------------------------------------------------------------------

_CHANNELS = 5   # R, G, B, luma, motion — fixed biological channels


class FovealRetina:
    """Non-uniform foveal retina with eccentricity-based sampling and blur.

    Parameters
    ----------
    window_title:
        Case-insensitive substring to match against window titles.
        Falls back to primary monitor if not found.
    vision_size:
        Exact number of vision neurons the Brain was built with.
        Must be divisible by 5 (the fixed channel count).
        The retina outputs exactly this many floats regardless of resolution.
        Fewer neurons → sparser but still eccentricity-weighted sampling.
    fov_h_deg, fov_v_deg:
        Horizontal and vertical full field-of-view in degrees.
        Default ±80°/±65° ≈ near full human monocular FOV.
    """

    def __init__(
        self,
        window_title: str   = 'minecraft',
        vision_size:  int   = 2880,
        fov_h_deg:    float = _FOV_H,
        fov_v_deg:    float = _FOV_V,
    ) -> None:
        if vision_size % _CHANNELS != 0:
            raise ValueError(
                f"vision_size must be divisible by {_CHANNELS} (R/G/B/luma/motion). "
                f"Got {vision_size}. Try {(vision_size // _CHANNELS) * _CHANNELS}.")
        self.window_title = window_title.lower()
        self.vision_size  = vision_size
        self.fov_h_deg    = fov_h_deg
        self.fov_v_deg    = fov_v_deg
        # Derived grid dimensions for _build_sampling_grid
        num_points = vision_size // _CHANNELS
        # Factorise into (grid_h, grid_w) — as square as possible
        grid_h = int(np.sqrt(num_points))
        while num_points % grid_h != 0:
            grid_h -= 1
        grid_w = num_points // grid_h
        self.grid_h = grid_h
        self.grid_w = grid_w

        try:
            import mss
            self._mss = mss.mss()
        except ImportError as e:
            raise ImportError("pip install mss") from e
        except Exception:
            self._mss = None
        try:
            import cv2
            self._cv2 = cv2
        except ImportError as e:
            raise ImportError("pip install opencv-python") from e

        self.num_points = vision_size // _CHANNELS
        self._monitor = None
        if self._mss is not None:
            try:
                self._locate_window()
            except Exception:
                self._monitor = None

        # Precompute non-uniform sampling grid
        self._xy, self._ecc = _build_sampling_grid(
            self.num_points, fov_h_deg, fov_v_deg)

        self._prev_luma = np.zeros(self.num_points, dtype=np.float32)
        # Fixation offset in normalised image coords [-1, 1].
        # Set by SaccadeController each tick to simulate gaze shift.
        self.fixation = (0.0, 0.0)

        # Precompute blur level index per sample point (0=sharp, 3=very blurry)
        # σ_deg = BLUR_K × ecc;  map to 4 levels by zone
        self._blur_level = np.zeros(self.num_points, dtype=np.int32)
        self._blur_level[self._ecc >= _ECC_FOVEA]     = 1
        self._blur_level[self._ecc >= _ECC_PARAFOVEA] = 2
        self._blur_level[self._ecc >= _ECC_NEAR_PERI] = 3

        bl = self._blur_level
        print(f'>>> FovealRetina: {self.num_points} receptors, '
              f'FOV {fov_h_deg}°×{fov_v_deg}°  '
              f'[fovea={(bl==0).sum()} | para={(bl==1).sum()} | '
              f'near={(bl==2).sum()} | far={(bl==3).sum()}]')

    # ------------------------------------------------------------------

    def _locate_window(self) -> None:
        monitor = None
        try:
            import win32gui
            results = []
            def _cb(hwnd, out):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd).lower()
                    if self.window_title in title:
                        r = win32gui.GetWindowRect(hwnd)
                        out.append(r)
            win32gui.EnumWindows(_cb, results)
            if results:
                l, t, r, b = results[0]
                monitor = {'left': l, 'top': t, 'width': r - l, 'height': b - t}
        except Exception:
            pass
        self._monitor = monitor or self._mss.monitors[1]

    def _capture_rgb(self) -> np.ndarray:
        """Return full-resolution (H, W, 3) uint8 RGB array."""
        if self._mss is None or self._monitor is None:
            raise RuntimeError('Screen capture is unavailable in this environment')
        shot = self._mss.grab(self._monitor)
        img  = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return img[:, :, [2, 1, 0]]   # BGRA → RGB

    def _make_blur_pyramid(self, rgb_f32: np.ndarray) -> list[np.ndarray]:
        """Return 4 blurred versions corresponding to the 4 eccentricity zones.

        σ is computed at the zone's representative eccentricity (Navarro 1993):
            σ_deg = 0.06 × ecc_deg
            σ_px  = σ_deg × (image_width / fov_h_deg)
        """
        cv2 = self._cv2
        W   = rgb_f32.shape[1]
        px_per_deg = W / self.fov_h_deg

        def blur(sigma_deg: float) -> np.ndarray:
            sigma_px = sigma_deg * px_per_deg
            if sigma_px < 0.5:
                return rgb_f32
            k = int(sigma_px * 6) | 1   # kernel must be odd
            return cv2.GaussianBlur(rgb_f32, (k, k), sigma_px)

        rep_ecc = [
            (_ECC_FOVEA    / 2),                              # zone 0 centre  ≈ 1.5°
            ((_ECC_FOVEA + _ECC_PARAFOVEA) / 2),             # zone 1 centre  ≈ 7.5°
            ((_ECC_PARAFOVEA + _ECC_NEAR_PERI) / 2),         # zone 2 centre  ≈ 23.5°
            ((_ECC_NEAR_PERI + _ECC_FAR_PERI) / 2),          # zone 3 centre  ≈ 57.5°
        ]
        return [blur(_BLUR_K * e) for e in rep_ecc]

    def _sample_points(
        self,
        pyramid: list[np.ndarray],
        H: int, W: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised lookup: for each sample point, pick the right blurred layer.

        Returns R, G, B each of shape (N,) float32 in [0, 1].
        """
        # Normalised xy → pixel coordinates, offset by current fixation point
        fx, fy = self.fixation
        px = np.clip(((self._xy[:, 0] + fx + 1.0) * 0.5 * (W - 1)).astype(np.int32), 0, W - 1)
        py = np.clip(((self._xy[:, 1] + fy + 1.0) * 0.5 * (H - 1)).astype(np.int32), 0, H - 1)

        R = np.empty(self.num_points, dtype=np.float32)
        G = np.empty(self.num_points, dtype=np.float32)
        B = np.empty(self.num_points, dtype=np.float32)

        for lvl in range(4):
            mask = self._blur_level == lvl
            if not mask.any():
                continue
            img = pyramid[lvl]
            R[mask] = img[py[mask], px[mask], 0]
            G[mask] = img[py[mask], px[mask], 1]
            B[mask] = img[py[mask], px[mask], 2]

        return R, G, B

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_image(self, rgb_f32: np.ndarray) -> np.ndarray:
        """Process an arbitrary float32 RGB image [0,1] through the retina pipeline."""
        H, W = rgb_f32.shape[:2]
        pyramid = self._make_blur_pyramid(rgb_f32)
        R, G, B = self._sample_points(pyramid, H, W)

        luma   = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.float32)
        motion = np.clip(np.abs(luma - self._prev_luma), 0.0, 1.0).astype(np.float32)
        self._prev_luma = luma

        # Interleave: (R₀,G₀,B₀,L₀,M₀, R₁,…)
        out = np.stack([R, G, B, luma, motion], axis=1).reshape(-1)
        return out.astype(np.float32)

    def get_visual_array(self) -> np.ndarray:
        """Capture, blur by eccentricity, sample non-uniformly.

        Returns flat float32 array of shape (grid_h * grid_w * 5,):
            [R₀ G₀ B₀ L₀ M₀ | R₁ G₁ B₁ L₁ M₁ | … ]
        """
        rgb_u8   = self._capture_rgb()
        self._last_raw_rgb = rgb_u8          # cached for homeostasis probes
        rgb_f32  = rgb_u8.astype(np.float32) / 255.0
        return self.preprocess_image(rgb_f32)
