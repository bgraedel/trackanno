# Measurements

What the suggestion model sees for each clip, how each value is defined, and how well it
matches ground truth. Lengths are in px, times in s, angles in rad.

## Beat waveform (`wave_*`)

The beat is measured as `processFile.m` of the flagella-analysis-pipeline (Fochler et al. 2025)
measures it, with the smoothing tolerance in px (the spatialScale^2 fix). Each sliding window
(`spectral.window` frames, 0.75 s by default) is one pipeline run, read at the clip's centre.

Per frame, the flagellum is a cubic spline through Base, Flag1..Flag5, Tip, smoothed as `spaps`
does with tolerance 0.2 L px^2, then moved so its base is the origin and the axis from the
Head-Base midpoint to the base is +x. Per window, L is the largest moving mean of the per-frame
length; frames shorter than 0.9 L are bad, gaps of up to two bad frames are bridged and the
longest run of good frames is kept. The flagellum is resampled to 200 points over 0.9 L, its
tangent angle psi(s, t) smoothed along s and Fourier transformed in time with no window
function.

| column | pipeline output | definition |
|---|---|---|
| `wave_freq_hz` | `domFreq` | the bin above 1 Hz whose amplitude, summed over +-1 Hz and along s, is largest |
| `wave_freq_proximal_hz`, `_central_hz`, `_distal_hz` | `domFreqProximal`, ... | the same over each third of 0.9 L |
| `wave_amp_mean`, `_asym`, `_para` | `legendreCoeffsDomFreqAmp` | Legendre coefficients 0 to 2 of that summed amplitude along s, at `wave_freq_hz` |
| `wave_amp_integral` | `domFreqAmpSummed` | the same amplitude integrated along s (rad px) |
| `wave_phase_asym`, `_para` | `legendreCoeffsDomFreqPhase` | Legendre coefficients 1 and 2 of the phase along s, unwrapped |
| `wave_phase_rsq` | `phaseLinearityRsq` | R^2 of a line through the phase |
| `wave_phase_cycles` | `numPhaseCycles` | range of the phase / 2 pi |
| `wave_static_mean` | `averageFlagAngleFromBase` | integral of the absolute time-mean angle, over L |
| `wave_freq_dominance` | `domFreqDominance` | the summed amplitude at `wave_freq_hz` over the whole spectrum less its static part |
| `wave_fit_error` | `qualityOfFitAngle` | relative RMS error of psi rebuilt from the bins within 1 Hz of 0 and of `wave_freq_hz` |
| `wave_xy_fit_error` | `qualityOfFitXYOptimal` | the same for the shape, each frame rotated to fit best |
| `wave_disp_amp_max` | `amplitudeSpatialAllData` | half the largest distance between two positions of a point, largest along s (px) |
| `wave_angular_amp` | `amplitudeAngularAllData` | half the range of psi over time, largest along s |
| `wave_length_px` | `flagLength` | L |
| `wave_bad_window` | `badFileFlag` | fewer than 30 good frames, or more than 20% bridged |

Read off the same analysis:

| column | definition |
|---|---|
| `wave_phase_slope` | slope of the line through the phase (rad/px) |
| `wave_wavelength_px`, `_ratio` | 2 pi / abs(slope), and that over L |
| `wave_direction` | -sign(slope): +1 base to tip, -1 tip to base |
| `wave_speed_px_s` | direction x wavelength x frequency |
| `wave_static_asym`, `_para` | Legendre coefficients 1 and 2 of the time-mean psi |
| `wave_disp_amp_mean`, `_tip` | the spatial amplitude averaged along s, and at the tip |
| `wave_excess` | the peak summed amplitude over the median one |
| `wave_censored`, `wave_resolvable_hz` | the peak is below the floor of 2.5 cycles per window (3.3 Hz at 0.75 s); the floor |
| `wave_interp_frac`, `wave_window_frames` | share of frames bridged; frames analysed |
| `wave_flow`, `_coherence`, `_direction` | direction from the correlation -<dpsi/dt dpsi/ds> of the keypoint chords, no spectrum |
| `wave_freq_drift_hz`, `wave_freq_track_sd_hz` | spread of the frequency nearby and over the whole track |

Details that matter for correctness:

* `spaps` is reproduced exactly: the cubic smoothing spline with trapezoid weights, its
  smoothing parameter solved for the residual budget. The tolerance is in px, so the same
  flagellum is smoothed relatively more at low magnification (see Limits).
* Ties in the spectrum go to the first bin, as MATLAB's `max` does, so a still flagellum reads
  1.3 Hz at 0.75 s.
* The pipeline takes the body centre from the body's skeleton; trackanno has Head and Base, so
  the centre is their midpoint. For a bent body the two axes differ by a constant angle, which
  shifts `wave_static_mean` and nothing else (0.07 rad median on the simulator's cells).
* A mirror image flips every angle; no column depends on which side the cell is seen from.

## Trajectory, body and shape

Read over the clip (`clips.frames`, 50 frames by default). Keypoint jitter at 100 fps is as large as
a cell's motion from one frame to the next, so every rate is a change over 0.1 s: the difference of
two consecutive 0.1 s means, divided by the time between them.

| columns | definition |
|---|---|
| `speed_*`, `immotile_frac` | box centre speed; immotile below 10 px/s (in px, so it depends on magnification) |
| `path_length`, `net_displacement`, `straightness`, `radius_gyration` | the box centre's path; straightness needs a mean move above 3x the 0.5 px jitter per 0.1 s |
| `turn_*` | change of the box centre's heading over 0.1 s, per second |
| `body_speed_*`, `body_immotile_frac`, `body_path_length`, `body_net_displacement`, `body_straightness`, `body_radius_gyration`, `body_turn_*` | the same for the Head-Base midpoint, which does not sway with the beat |
| `body_rotation_speed` | rotation of the Head-Base axis over 0.1 s, per second |
| `body_len_*` | Head to Base distance |
| `flagellum_len_*`, `_cv` | sum of circular arcs through Base, Flag1..Flag5, Tip |
| `tip_angle_std` | spread of the Base-to-Tip direction against the body axis |
| `shape_*`, `area_*`, `aspect_*` | keypoint cloud extent and principal axes; box area and aspect |

The box contains the flagellum, so it sways with the beat: prefer the `body_*` columns where there
are keypoints. The box columns are what a project without keypoints has.

## Validation

### Against the MATLAB pipeline

`validation/matlab/` runs the pipeline itself on simulated cells and compares every output it
writes with trackanno's:

    uv run python validation/matlab/cells.py --simulator PATH/synthetic_leishmania.py --out OUT
    uv run python validation/matlab/run_matlab.py --cells OUT --pipeline PATH/flagella-analysis-pipeline
    uv run python validation/matlab/compare.py --cells OUT

60 simulated cells, two 0.75 s snippets each (100 fps, 0.325 um/px), 90 of them with a
resolvable beat:

* **Same input**: MATLAB reads the keypoints trackanno reads, written as a trace. This tests the
  analysis itself.
* **Dense skeleton**: MATLAB reads the simulator's whole skeleton at 1 px (clean) or on integer
  pixels (realistic); trackanno reads the keypoints, clean or jittered. This adds what eight
  keypoints cannot carry.

Share of snippets within 1% of MATLAB's value (of its `wave_amp_mean` for `wave_amp_asym` and
`wave_amp_para`, of pi for `wave_phase_para`):

| column | same input | same input, jitter | dense, clean | dense, realistic |
|---|---|---|---|---|
| `wave_freq_hz` | 100% | 98% | 98% | 81% |
| `wave_freq_proximal_hz` | 100% | 98% | 97% | 76% |
| `wave_freq_central_hz` | 100% | 98% | 98% | 81% |
| `wave_freq_distal_hz` | 100% | 97% | 98% | 82% |
| `wave_amp_mean` | 90% | 86% | 44% | 7% |
| `wave_amp_asym` | 84% | 84% | 29% | 4% |
| `wave_amp_para` | 77% | 74% | 32% | 3% |
| `wave_amp_integral` | 90% | 86% | 40% | 16% |
| `wave_phase_asym` | 84% | 81% | 22% | 13% |
| `wave_phase_para` | 96% | 92% | 58% | 20% |
| `wave_phase_rsq` | 93% | 87% | 68% | 52% |
| `wave_phase_cycles` | 63% | 68% | 20% | 9% |
| `wave_static_mean` | 100% | 93% | 6% | 7% |
| `wave_freq_dominance` | 92% | 88% | 20% | 0% |
| `wave_fit_error` | 82% | 86% | 12% | 3% |
| `wave_xy_fit_error` | 77% | 81% | 28% | 2% |
| `wave_disp_amp_max` | 68% | 47% | 28% | 2% |
| `wave_angular_amp` | 66% | 71% | 27% | 4% |
| `wave_length_px` | 100% | 100% | 86% | 8% |
| `wave_bad_window` | 100% agree | 100% agree | 100% agree | 93% agree |

On the same input every column correlates at r >= 0.97 with MATLAB's, except `wave_phase_rsq`
on clean keypoints (0.68, from one window; 93% within 1%). What remains is sampling: the pipeline
smooths at the trace's own points, about 1 px apart, and trackanno at 128 points; sampling at
1 px halves the differences. Against the dense skeleton, the static angle differs by the body
axis (see above), the amplitude profile carries detail between keypoints that the spline cannot
recover, and on realistic input the integer-pixel skeleton and the keypoint jitter each add noise
of their own.

### Against the simulator

`validation/run.py` tracks random cells from the synthetic Leishmania simulator as keypoints, with
the jitter measured on real data (0.3 px on Head and Base, 0.6 px along the flagellum), 3% dropped
frames, 0.5% dropped keypoints and random detector confidences. Each beat column is compared with
the same analysis run on the simulator's exact flagellum over the same frames, smoothing
included; frequency, wavelength, direction and length are also compared with the simulator's
own parameters.

    uv run python validation/run.py --simulator PATH/synthetic_leishmania.py --out OUT --workers 6

Results (250 cells, 4835 windows of 0.75 s with a resolvable beat, 100 fps, 0.325 um/px; p90 of
the absolute error, relative where marked %):

| column | clean keypoints | with jitter and dropouts |
|---|---|---|
| `wave_freq_hz` | 0% | 5.9% (one bin is 1.3 Hz) |
| `wave_amp_mean`, `wave_amp_integral` | 12%, 12% | 26%, 27% (median -4.7%, -2.8%) |
| `wave_amp_asym`, `wave_amp_para` | 0.06, 0.10 | 0.08, 0.12 |
| `wave_phase_asym`, `wave_phase_para` | 0.33, 0.29 rad | 0.45, 0.35 rad |
| `wave_phase_rsq`, `wave_phase_cycles` | 0.03, 0.11 | 0.12, 0.16 |
| `wave_static_mean` | 0.018 rad | 0.037 rad |
| `wave_freq_dominance`, `wave_fit_error`, `wave_xy_fit_error` | 0.02, 0.02, 0.007 | 0.17, 0.20, 0.035 |
| `wave_disp_amp_max`, `wave_angular_amp` | 41%, 19% | 110%, 114% (median +26%, +20%) |
| `wave_length_px` | 1.7% | 5.8% |
| `wave_bad_window` | 99.8% agree | 91% agree |
| `wave_direction` (against the simulator) | 94% right | 97.5% right |
| `wave_flow_direction` | 99.9% right | 94% right |

Medians of the signed errors are within 2% (0.05 rad for angles) unless given; jitter also raises
`wave_fit_error` and lowers `wave_freq_dominance` by 0.05 (median). Clean keypoints miss only what
six segments cannot carry; jitter mainly inflates the amplitudes taken over all frames
(`wave_disp_amp_*`, `wave_angular_amp`) and marks short flagella bad (see Limits).
`wave_excess` is the confidence of a window: at 8 or more (56% of beating windows) the frequency
is within one bin of the simulator's in 99.7% and the direction right in 96%; below 3 (15%)
neither is better than chance. No window of a still cell reaches 8.

Trajectory and shape over the same cells (50-frame clips, p90 of the absolute error with jitter).
Motion is compared with the same estimator on the simulator's exact body centre and axis, the
flagellum length with the simulator's own, and keypoint descriptors with the noise-free keypoints:

| column | error |
|---|---|
| `body_speed_mean`, `body_speed_median` | 1.0 px/s |
| `body_rotation_speed` | 0.03 rad/s |
| `body_turn_abs_mean` | 0.6 rad/s (the noise floor of a heading over 0.1 s) |
| `body_net_displacement`, `body_path_length` | 0.5 px |
| `body_straightness`, `straightness` | 0.03, 0.05 |
| `body_immotile_frac`, `immotile_frac` | 0.13, 0.10 |
| `flagellum_len_mean` | 9.5% (median -0.1%; the tail is waves too short to resolve) |
| `body_len_mean`, `shape_extent_mean` | 2.5%, 0.4% |
| `tip_angle_std` | 0.02 rad |

### Real tracks

On 66 real tracks (powa_6, 100 fps, 0.325 um/px), measured windows above 15 Hz are 100% tip to base
and windows of 4 to 10 Hz 81% base to tip with a static angle of 0.87 rad, as expected for
*Leishmania*; the spectral and flow directions agree on 99% and 93% of them. 38% of real windows
are marked bad.

## Limits

* **Spatial resolution.** Seven nodes give six segments, so a wave shorter than 2.5 segments
  (lambda < 0.42 L) cannot be measured: it folds into a longer wave running the other way, with a
  linear phase that looks entirely normal. Direction is then right in fewer than half the windows.
  Treat a measured `wave_wavelength_ratio` below 0.5 with suspicion.
* **Slow beats.** A beat needs 2.5 cycles in the window; slower ones are censored. A longer
  `spectral.window` lowers the floor but blurs switches between beat types.
* **Smoothing at low magnification.** The smoothing allows an RMS residual of 0.45 px at any
  scale, so a flagellum of 25 to 90 px (20x) is smoothed hard. Against the simulator's parameters,
  wavelengths come out 31% long (median), and on the exact flagellum the pipeline's direction is
  wrong in 7% of windows, mostly flagella under 30 px or waves longer than 2 L. The pipeline does
  the same on such movies.
* **Keypoint jitter.** Read `wave_excess` as the confidence of a window; below 3 the beat is lost
  in the noise. Jitter moves the per-frame length of a short flagellum by more than the 10% the
  pipeline allows, so a quarter of windows on flagella under 30 px are marked bad (3% above that).
  Turning below about 0.5 rad/s cannot be told from straight swimming over 0.1 s.
* **Units.** Lengths and speeds are in px, so features from movies at different magnifications are
  not comparable, and `IMMOTILE_PX_PER_SEC` (10 px/s) assumes about 0.3 um/px.
