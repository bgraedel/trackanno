"""Curvature kymographs from tracked keypoints.

Signed curvature along the arclength against time: a travelling wave is diagonal
stripes, a standing wave vertical ones, a mistracked frame a white column.
Curvature is intrinsic, so the crop's own pixels serve directly with no
stabilisation or body-axis alignment. Being a second derivative it comes from a
smoothing spline fitted through the chain once per frame and differentiated
analytically; finite differences on seven noisy points are almost pure noise.

Node convention, matching :mod:`trackanno.clipfeatures`: node 0 is Head, node 1
is Base, nodes 1..N the flagellum base to tip, arclength ``s`` 0 at the base to
1 at the tip. Positive curvature bends anticlockwise in the image coordinate
system walking base to tip - image y points *down*, so clockwise on screen. The
sign flips under a mirror; only the relative sign matters for reading a beat,
but it is fixed so two clips can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Kymograph", "curvature_kymograph", "resample_centreline", "chain_from_edges",
           "SignVerdict", "canonical_sign", "SIGN_MODES"]

SIGN_MODES = ("raw", "canonical", "absolute")

# How much static curvature there has to be, relative to the beat, before its
# sign is worth believing. On real trypanosome clips the ratio runs 0.13 to 1.13
# (median 0.23), so this decides for perhaps half of them; the split-half test
# below keeps the other half honest.
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
    """Decide a consistent handedness for one clip's curvature field.

    A 2D image cannot tell a cell from its own mirror image: the two produce
    curvature fields differing by an overall factor of -1 (20 real clips split
    11 to 9). Every EVEN functional of curvature is mirror-invariant and so
    useless here - amplitude profile, power spectrum, wave speed, direction of
    travel. Only an ODD one separates them, and the physical one is the static
    mode: the time-average ``psi_0(s)``, the intrinsic curvature left behind
    once averaging over time cancels the beat.

    So: flip the clip so its static mode is positive, guarded twice because on a
    near-symmetric beater that mode is small and its sign is noise. It must be
    big enough relative to the beat (``min_ratio``), and the two halves of the
    clip must independently agree. When either fails the verdict is ``+1`` with
    ``confident=False``, left exactly as measured rather than flipped on a coin
    toss, and the caller is expected to say so.
    """
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

    # Weighted by its own magnitude so the decision comes from where the static
    # bend actually is, not from a long tail of near-zero samples that happen to
    # share a sign.
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


def _unwrap_in_time(psi):
    """Finish the unwrap along the time axis, and remove whole-cell rotation.

    Unwrap along arclength first and only then along time, as
    ``wavespectra.spectral_series`` does: the node where the wave crosses +-pi
    MOVES as the wave travels, so a fixed arclength position jumps by 2*pi
    between frames. Subtracting the per-frame mean then removes the cell's
    absolute heading, which drifts as it swims and would dominate the plot,
    leaving the shape rotation invariant as curvature already is.
    """
    field = np.asarray(psi, dtype=np.float64)
    usable = np.isfinite(field).all(axis=1)
    if usable.sum() < 2:
        return psi
    block = np.unwrap(field[usable], axis=0)
    block = block - block.mean(axis=1, keepdims=True)
    out = np.full(field.shape, np.nan, dtype=np.float32)
    out[usable] = block.astype(np.float32)
    return out


def chain_from_edges(n_nodes: int, edges) -> list:
    """The keypoint indices along the flagellum, taken from the skeleton.

    The order decides which points are joined into a curve and fixes the SIGN of
    the curvature, since walking a bend from the other end negates it. Read from
    ``keypoints.edges`` rather than guessed: the longest path through that graph
    is the flagellum, and a head hanging off the base drops out as a side
    branch. Returns [] when the skeleton says nothing usable, leaving the caller
    to fall back to plain node order and say so.
    """
    adjacency: dict[int, set] = {}
    for a, b in edges or []:
        a, b = int(a), int(b)
        if a == b or not (0 <= a < n_nodes and 0 <= b < n_nodes):
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    if len(adjacency) < MIN_NODES:
        return []

    # Skeletons are a handful of nodes, so an exhaustive search for the longest
    # simple path is affordable and exact; the tree-diameter trick would be
    # wrong the moment the skeleton contains a cycle.
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

    # Two traversals of one path are equally long; take the one that starts at
    # the lower node index so a given config always gives the same sign.
    if best and best[0] > best[-1]:
        best.reverse()
    return best if len(best) >= MIN_NODES else []

# Below this many tracked points a curve cannot be fitted at all: a cubic needs
# four, and dropping to a lower order silently changes what curvature means.
MIN_NODES = 4

# Positional noise the spline is fitted against, in pixels: how hard the fit may
# pull away from the measured keypoints, and the knob that matters most here.
# Too little and the second derivative amplifies tracking jitter into stripes
# that are not there; too much and a real beat is flattened into a smear.
DEFAULT_SMOOTH_PX = 0.6


@dataclass
class Kymograph:
    """Signed curvature on a (time x arclength) grid."""

    kappa: np.ndarray          # (T, S) float32, 1/px; NaN where unfittable
    length_px: np.ndarray      # (T,) float32 arclength of the chain per frame
    n_nodes: int               # tracked points the curve was fitted through
    smooth_px: float
    psi: np.ndarray | None = None   # (T, S) tangent angle, unwrapped in s and t

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
        """A symmetric colour limit that one bad frame cannot blow out.

        Tracking failures land at the extremes of a curvature field, so scaling
        to max() hands the whole colour range to the worst frame.
        """
        finite = self.kappa[np.isfinite(self.kappa)]
        if not finite.size:
            return 1.0
        limit = float(np.percentile(np.abs(finite), percentile))
        return limit if limit > 1e-9 else 1.0

    def with_sign(self, mode: str = "canonical", min_ratio: float = SIGN_MIN_RATIO):
        """(kymograph, verdict) under one of the sign conventions.

        ``raw`` is what was measured. ``canonical`` flips the whole clip so its
        static curvature is positive, when that is decidable. ``absolute`` drops
        the sign, merging a leftward bend with a rightward one - fine when only
        the magnitude matters, bad for reading a travelling wave. The flip is a
        single factor for the WHOLE clip; choosing per frame would destroy the
        very thing the plot shows, a pattern whose sign alternates in order.
        """
        if mode == "absolute":
            flipped = Kymograph(np.abs(self.kappa), self.length_px, self.n_nodes,
                                self.smooth_px)
            return flipped, SignVerdict(1, True, 0.0, "magnitude only, sign discarded")
        if mode != "canonical":
            return self, SignVerdict(1, True, 0.0, "as measured")

        verdict = canonical_sign(self.kappa, min_ratio)
        if verdict.sign == 1:
            return self, verdict
        return (Kymograph(-self.kappa, self.length_px, self.n_nodes, self.smooth_px),
                verdict)

    def normalised(self) -> np.ndarray:
        """Curvature made dimensionless as ``kappa * L``.

        1/px cannot be compared between a long cell and a short one, or across
        magnifications; multiplying by the flagellum's own length gives total
        turning, which can.
        """
        length = self.length_px[:, None]
        with np.errstate(invalid="ignore"):
            return (self.kappa * length).astype(np.float32)


def resample_centreline(x, y, n_samples: int = 128, smooth_px: float = DEFAULT_SMOOTH_PX):
    """Fit one frame's chain and resample it at equal arclength.

    Returns ``(points, kappa, length, psi)``: points is (n_samples, 2), kappa is
    (n_samples,) signed curvature in 1/px, length is the arclength in px, and psi
    is the tangent angle in radians, unwrapped along the arclength. Returns
    ``(None, None, nan, None)`` when the chain cannot carry a curve.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < MIN_NODES:
        return None, None, float("nan"), None

    # Duplicate points make the spline parametrisation singular, and a tracker
    # that loses a node often reports it at the previous node's position.
    step = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate([[True], step > 1e-6])
    x, y = x[keep], y[keep]
    if x.size < MIN_NODES:
        return None, None, float("nan"), None

    import warnings

    from scipy.interpolate import splev, splprep

    # s bounds the SUM of squared residuals and so scales with the number of
    # points; a per-point pixel tolerance keeps the knob's meaning the same
    # whether the model has 6 keypoints or 30.
    total = float(x.size) * float(smooth_px) ** 2
    try:
        with warnings.catch_warnings():
            # "s too small" only means the fit stopped at the iteration cap and
            # is closer to the data than asked; still a valid smoothing spline,
            # and it happens on every badly tracked frame.
            warnings.simplefilter("ignore", RuntimeWarning)
            tck, _ = splprep([x, y], s=total, k=3)
    except (TypeError, ValueError):
        return None, None, float("nan"), None

    # Equal arclength, not equal parameter: splprep parametrises by chord
    # length, which bunches samples exactly where the curve bends and would
    # distort every feature read off the result.
    dense = np.linspace(0.0, 1.0, max(16 * n_samples, 256))
    dx, dy = splev(dense, tck, der=1)
    speed = np.hypot(dx, dy)
    arc = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2.0 * np.diff(dense))])
    length = float(arc[-1])
    if not np.isfinite(length) or length <= 0:
        return None, None, float("nan"), None

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

    # The tangent angle, from the same analytic derivatives, unwrapped ALONG THE
    # ARCLENGTH: atan2 wraps at +-pi, so a flagellum curling past that shows a
    # 2*pi cliff at whichever node crosses it. Unwrapping along time is the
    # caller's job, needing every frame at once. Same quantity as the curvature
    # above (kappa = dpsi/ds, agreeing to 0.5%), kept because psi is
    # FIRST-order and reads more cleanly on a noisy track.
    psi = np.unwrap(np.arctan2(d1y, d1x))
    return np.column_stack([px, py]), kappa, length, psi


def curvature_kymograph(
    kx, ky, n_samples: int = 128, smooth_px: float = DEFAULT_SMOOTH_PX,
    order=None, first_node: int = 1,
) -> Kymograph:
    """Signed curvature along the flagellum for every frame of a clip.

    ``kx``/``ky`` are (frames, nodes) keypoint coordinates in any single
    consistent space - the crop's own pixels are the natural one. ``order`` is
    the keypoint indices along the flagellum, base first; pass the result of
    :func:`chain_from_edges` so the traversal comes from the project's own
    skeleton. Without it the fallback is nodes ``first_node..N``, only a
    convention, matching Head/Base/... in :mod:`trackanno.clipfeatures`.
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
    psi = np.full((n_frames, n_samples), np.nan, dtype=np.float32)
    lengths = np.full(n_frames, np.nan, dtype=np.float32)
    for t in range(n_frames):
        _, k, length, angle = resample_centreline(
            chain_x[t], chain_y[t], n_samples, smooth_px)
        if k is not None:
            kappa[t] = k.astype(np.float32)
            psi[t] = angle.astype(np.float32)
            lengths[t] = length

    return Kymograph(
        kappa=kappa, length_px=lengths,
        n_nodes=int(chain_x.shape[1]), smooth_px=float(smooth_px),
        psi=_unwrap_in_time(psi),
    )


def kymograph_from_boxes(boxes, n_kpts: int, n_frames: int, min_conf: float = 0.0,
                         **kwargs) -> Kymograph:
    """Build one from a clip's box table, which is how the GUI has the data.

    One row per (frame, track): only the target track carries the flagellum, and
    a frame the tracker missed has no row, so the grid is filled with NaN first
    and written by frame index, never by row order.
    """
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

    # A keypoint the model was not confident about is worse than a missing one:
    # it drags the spline somewhere the flagellum is not. Dropping it lets the
    # fit span the gap instead.
    conf_columns = [f"kp_c_{k}" for k in range(n_kpts)]
    if min_conf > 0 and all(c in target.columns for c in conf_columns):
        conf = np.full((n_frames, n_kpts), np.inf)
        for k, column in enumerate(conf_columns):
            conf[frames, k] = target[column].to_numpy(dtype=float)[inside]
        weak = conf < min_conf
        kx[weak], ky[weak] = np.nan, np.nan
    return curvature_kymograph(kx, ky, **kwargs)
