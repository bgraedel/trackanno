"""Curvature kymographs: signed curvature (1/px) along the flagellum against time.

Curvature comes from a smoothing spline per frame; arclength runs 0 at the base to 1 at the tip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Kymograph", "curvature_kymograph", "resample_centreline", "chain_from_edges",
           "SignVerdict", "canonical_sign", "SIGN_MODES"]

SIGN_MODES = ("raw", "canonical")

# Minimum static-to-beat curvature ratio for the static sign to count.
SIGN_MIN_RATIO = 0.20


@dataclass
class SignVerdict:
    """Which way up a clip should be drawn, and whether that is knowable."""

    sign: int                  # +1 or -1, the factor to multiply curvature by
    confident: bool
    ratio: float               # ||static mode|| / ||beat amplitude||
    reason: str

    @property
    def flipped(self) -> bool:
        return self.sign < 0


def canonical_sign(kappa, min_ratio: float = SIGN_MIN_RATIO) -> SignVerdict:
    """Sign making the clip's static curvature positive, when large and stable enough."""
    field = np.asarray(kappa, dtype=np.float64)
    if field.ndim != 2 or not field.size or not np.isfinite(field).any():
        return SignVerdict(1, False, 0.0, "no curvature to judge")

    def static_of(block):
        with np.errstate(invalid="ignore"):
            static = np.nanmean(block, axis=0)
            beat = np.nanstd(block, axis=0)
        return static, beat

    static, beat = static_of(field)
    if not np.isfinite(static).any():
        return SignVerdict(1, False, 0.0, "no curvature to judge")
    static = np.nan_to_num(static)
    beat = np.nan_to_num(beat)
    norm_static = float(np.linalg.norm(static))
    norm_beat = float(np.linalg.norm(beat))
    ratio = norm_static / norm_beat if norm_beat > 1e-12 else float("inf")

    # Weighted by magnitude so near-zero samples do not decide.
    total = float(np.sum(static * np.abs(static)))
    sign = -1 if total < 0 else 1

    if ratio < min_ratio:
        return SignVerdict(
            1, False, ratio,
            f"the static curvature is only {ratio:.0%} of the beat, so its sign "
            "is not distinguishable from noise",
        )

    half = field.shape[0] // 2
    if half >= 4:
        first, _ = static_of(field[:half])
        second, _ = static_of(field[half:])
        first, second = np.nan_to_num(first), np.nan_to_num(second)
        sign_a = -1 if float(np.sum(first * np.abs(first))) < 0 else 1
        sign_b = -1 if float(np.sum(second * np.abs(second))) < 0 else 1
        if sign_a != sign_b:
            return SignVerdict(
                1, False, ratio,
                "the two halves of the clip disagree about which way it bends",
            )

    return SignVerdict(
        sign, True, ratio,
        ("flipped so the static curvature is positive" if sign < 0
         else "already the right way up"),
    )


def chain_from_edges(n_nodes: int, edges) -> list:
    """Keypoint indices along the flagellum: the longest path through the skeleton, or []."""
    adjacency: dict[int, set] = {}
    for a, b in edges or []:
        a, b = int(a), int(b)
        if a == b or not (0 <= a < n_nodes and 0 <= b < n_nodes):
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    if len(adjacency) < MIN_NODES:
        return []

    # Exhaustive longest-simple-path search; skeletons are small.
    best: list = []

    def walk(node: int, seen: list) -> None:
        nonlocal best
        if len(seen) > len(best):
            best = list(seen)
        for neighbour in sorted(adjacency[node]):
            if neighbour not in seen:
                seen.append(neighbour)
                walk(neighbour, seen)
                seen.pop()

    for start in sorted(adjacency):
        walk(start, [start])

    # Start at the lower node index so the sign is deterministic.
    if best and best[0] > best[-1]:
        best.reverse()
    return best if len(best) >= MIN_NODES else []


# A cubic spline needs four points.
MIN_NODES = 4

# Per-point positional tolerance of the spline fit, px.
DEFAULT_SMOOTH_PX = 0.6
# Equal-arclength sites a dense centerline is smoothed on.
DENSE_SITES = 256


@dataclass
class Kymograph:
    """Signed curvature on a (time x arclength) grid."""

    kappa: np.ndarray          # (T, S) float32, 1/px; NaN where unfittable
    length_px: np.ndarray      # (T,) float32 arclength of the chain per frame
    n_nodes: int               # tracked points the curve was fitted through
    smooth_px: float

    @property
    def n_frames(self) -> int:
        return int(self.kappa.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.kappa.shape[1])

    @property
    def usable(self) -> bool:
        return bool(self.kappa.size) and bool(np.isfinite(self.kappa).any())

    def scale(self, percentile: float = 98.0) -> float:
        """A symmetric colour limit at a percentile of absolute curvature."""
        finite = self.kappa[np.isfinite(self.kappa)]
        if not finite.size:
            return 1.0
        limit = float(np.percentile(np.abs(finite), percentile))
        return limit if limit > 1e-9 else 1.0

    def with_sign(self, mode: str = "canonical", min_ratio: float = SIGN_MIN_RATIO):
        """(kymograph, verdict): ``raw`` as measured, ``canonical`` flipped per clip."""
        if mode != "canonical":
            return self, SignVerdict(1, True, 0.0, "as measured")

        verdict = canonical_sign(self.kappa, min_ratio)
        if verdict.sign == 1:
            return self, verdict
        return (Kymograph(-self.kappa, self.length_px, self.n_nodes, self.smooth_px),
                verdict)

    def normalised(self) -> np.ndarray:
        """Dimensionless curvature ``kappa * L``, with L the flagellum length."""
        length = self.length_px[:, None]
        with np.errstate(invalid="ignore"):
            return (self.kappa * length).astype(np.float32)


def resample_centreline(x, y, n_samples: int = 128, smooth_px: float = DEFAULT_SMOOTH_PX,
                        cutoff_px: float | None = None):
    """Fit one frame's chain and resample at equal arclength.

    ``smooth_px`` is the spline's tolerance per point, for a few keypoints. A dense centerline
    takes ``cutoff_px`` instead: it is smoothed over that distance, however densely it is sampled.
    Returns ``(points, kappa 1/px, length px)``, or ``(None, None, nan)`` if unfittable.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < MIN_NODES:
        return None, None, float("nan")

    # Duplicate points make the spline singular.
    step = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate([[True], step > 1e-6])
    x, y = x[keep], y[keep]
    if x.size < MIN_NODES:
        return None, None, float("nan")

    import warnings

    from scipy.interpolate import splev, splprep

    # splprep's s bounds the summed squared residuals, so scale it by point count.
    total = float(x.size) * float(smooth_px) ** 2
    if cutoff_px is not None:
        x, y = _smooth_over(x, y, float(cutoff_px))
        total = 0.0
    try:
        with warnings.catch_warnings():
            # "s too small" still yields a valid spline.
            warnings.simplefilter("ignore", RuntimeWarning)
            tck, _ = splprep([x, y], s=total, k=3)
    except (TypeError, ValueError):
        return None, None, float("nan")

    # Resample by arclength, not by spline parameter.
    dense = np.linspace(0.0, 1.0, max(16 * n_samples, 256))
    dx, dy = splev(dense, tck, der=1)
    speed = np.hypot(dx, dy)
    arc = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2.0 * np.diff(dense))])
    length = float(arc[-1])
    if not np.isfinite(length) or length <= 0:
        return None, None, float("nan")

    wanted = np.linspace(0.0, length, n_samples)
    u = np.interp(wanted, arc, dense)

    px, py = splev(u, tck, der=0)
    d1x, d1y = splev(u, tck, der=1)
    d2x, d2y = splev(u, tck, der=2)
    # Parametrisation-invariant, so it does not matter that u is not arclength.
    denominator = (d1x * d1x + d1y * d1y) ** 1.5
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = (d1x * d2y - d1y * d2x) / denominator
    kappa[~np.isfinite(kappa)] = np.nan
    return np.column_stack([px, py]), kappa, length


def _smooth_over(x: np.ndarray, y: np.ndarray, cutoff_px: float) -> tuple[np.ndarray, np.ndarray]:
    """The polyline at equal arclength, smoothed so a bend ``cutoff_px`` long keeps half its size."""
    from .wavespectra import linear_smooth

    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    at = np.linspace(0.0, arc[-1], DENSE_SITES)
    rows = np.vstack([np.interp(at, arc, x), np.interp(at, arc, y)])
    smoothed = linear_smooth(rows, cutoff_px * (DENSE_SITES - 1) / max(arc[-1], 1e-9))
    return smoothed[0], smoothed[1]


def curvature_kymograph(
    kx, ky, n_samples: int = 128, smooth_px: float = DEFAULT_SMOOTH_PX,
    order=None, first_node: int = 1, cutoff_px: float | None = None,
) -> Kymograph:
    """Signed curvature along the flagellum for every frame of (frames, nodes) keypoints.

    ``order`` is the base-first chain from :func:`chain_from_edges`; else ``first_node..N``.
    ``cutoff_px`` smooths a dense centerline over a distance (see :func:`resample_centreline`).
    """
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    if kx.ndim != 2 or kx.shape != ky.shape:
        raise ValueError(f"kx and ky must be matching (frames, nodes); got {kx.shape} and {ky.shape}")

    if order:
        picked = [i for i in order if 0 <= int(i) < kx.shape[1]]
        chain_x, chain_y = kx[:, picked], ky[:, picked]
    else:
        chain_x = kx[:, first_node:]
        chain_y = ky[:, first_node:]
    n_frames = chain_x.shape[0]
    n_samples = max(8, int(n_samples))

    kappa = np.full((n_frames, n_samples), np.nan, dtype=np.float32)
    lengths = np.full(n_frames, np.nan, dtype=np.float32)
    for t in range(n_frames):
        _, k, length = resample_centreline(chain_x[t], chain_y[t], n_samples, smooth_px, cutoff_px)
        if k is not None:
            kappa[t] = k.astype(np.float32)
            lengths[t] = length

    return Kymograph(
        kappa=kappa, length_px=lengths,
        n_nodes=int(chain_x.shape[1]), smooth_px=float(smooth_px),
    )


def kymograph_from_boxes(boxes, n_kpts: int, n_frames: int, min_conf: float = 0.0,
                         **kwargs) -> Kymograph:
    """Kymograph of the target track in a clip's box table, NaN on missing frames."""
    kx = np.full((n_frames, n_kpts), np.nan)
    ky = np.full((n_frames, n_kpts), np.nan)
    if boxes is None or len(boxes) == 0 or n_kpts <= 0:
        return curvature_kymograph(kx, ky, **kwargs)

    target = boxes[boxes["is_target"]] if "is_target" in boxes.columns else boxes
    if target.empty:
        return curvature_kymograph(kx, ky, **kwargs)

    frames = target["t"].to_numpy(dtype=int)
    inside = (frames >= 0) & (frames < n_frames)
    frames = frames[inside]
    for k in range(n_kpts):
        cx, cy = f"kp_x_{k}", f"kp_y_{k}"
        if cx not in target.columns or cy not in target.columns:
            continue
        kx[frames, k] = target[cx].to_numpy(dtype=float)[inside]
        ky[frames, k] = target[cy].to_numpy(dtype=float)[inside]

    # Drop low-confidence keypoints; the spline spans the gap.
    conf_columns = [f"kp_c_{k}" for k in range(n_kpts)]
    if min_conf > 0 and all(c in target.columns for c in conf_columns):
        conf = np.full((n_frames, n_kpts), np.inf)
        for k, column in enumerate(conf_columns):
            conf[frames, k] = target[column].to_numpy(dtype=float)[inside]
        weak = conf < min_conf
        kx[weak], ky[weak] = np.nan, np.nan
    return curvature_kymograph(kx, ky, **kwargs)
