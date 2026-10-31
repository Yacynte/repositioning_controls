# Drone repositioning: controls and evaluation

Python side of a monocular **return-to-location** system for a drone. A target image is captured at a
reference pose. Later, the drone starts from a pose near the target and flies back using only its
camera image and the stored target image.

The image matching, rotation estimation (PNEC) and control run in the main C++/Python pipeline. This
folder contains the drone/controller interface scripts and the **evaluation** of the runs, both in
simulation (Unreal Engine) and on the real drone.

Results:

- `finalMotion/RESULTS.md`: main simulation results. 13 configurations × 18 trials: clear weather,
  rain, snow, fog, and ablations of the settle threshold, control law, rotation estimator and
  translation source.
- [data/motionLog5_03/RESULTS1.md](data/motionLog5_03/RESULTS1.md): an earlier clear-weather PNEC
  run (6 trials).

---

## Repository contents

| file | purpose |
|---|---|
| **`analyze_final_motion.py`** | **Main evaluation script** for simulation and drone runs: pose accuracy, statistics, image comparison and all figures. |
| `repositioning_score.py` | Image comparison used by the evaluation: DINOv2 global similarity + SuperPoint/LightGlue matching (overlap, alignment, pixel error). Also usable on its own for one target/start/final triple. |
| `rt_ambiguity_analysis.py` | Geometry helpers used by the evaluation: camera poses from `meta.json`, depth loading, projection, rotation–translation ambiguity. |
| `clip_comparison.py`, `evaluate_repositioning.py`, `pose_observability.py`, `plot_drone_runs.py` | Earlier analysis scripts (CLIP comparison, compensation verdict, Fisher-information analysis, separate drone plots). Superseded by `analyze_final_motion.py`; kept for reference. |
| `controls.py`, `main.py`, `main_wsl.py`, `new_main.py` | UDP/RTSP interface to the drone and the controller (sending commands, keyboard control). |
| `plot_result.py` | Plots a ground-truth MotionLog against an algorithm log. |
| `stop_detector.py` | Experimental GRU classifier for the per-axis stop decision. |
| `test.py`, `test2.py` | Manual keyboard-control tests. |

---

## Setup

Python 3.12 or newer (developed with 3.13).

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs CLIP and LightGlue from GitHub, so `git` must be available. The first
image comparison downloads the DINOv2 weights through `torch.hub` (about 350 MB) and the
SuperPoint/LightGlue weights. A GPU is used automatically if PyTorch finds one; on CPU the image
comparison takes most of the run time.

---

## Data

The raw data is **not in the repository** (depth maps alone are about 60 MB per snapshot). Only the
results are versioned: the RESULTS documents, `results.json`, `summary.*`, `trials.csv`,
`drone_runs.csv`, `image_scores.json`, the summary and drone figures, and the GT4 target with its
per-trial figures as an example. To reproduce the results, place the raw data in `finalMotion/`:

```
finalMotion/
├── GroundTruth/                         # simulation targets
│   └── GT<k>_<YYYYmmdd_HHMMSS>/          # one per target position, k = 1..6
│       ├── rgb.png                       # target image
│       ├── meta.json                     # exact camera pose + intrinsics
│       ├── depth_planar_cm.f32           # depth map (float32, HxW, cm)
│       ├── world_xyz_cm.f64              # world position per pixel (float64, HxWx3, cm)
│       └── MotionLog.csv                 # drone pose at the target
├── logData<set>/                         # simulation runs, one folder per configuration
│   ├── config_<ts>.txt                   # ablation config, incl. target=GroundTruth/GT<k>_.../rgb.png
│   ├── log_<ts>.txt                      # controller log ("Maintained target position" / "Timed Out")
│   ├── dataLog_<ts>.csv                  # per-frame estimated errors
│   ├── matches_<ts>.npz                  # last feature matches
│   └── images/imageStart<ts>.png, imageEnd<ts>.png
├── motion<set>/                          # same <set> suffix as logData<set>
│   └── Final<n>_<YYYYmmdd_HHMMSS>/       # snapshot at the end of each trial
│       ├── meta.json                     # true final camera pose
│       ├── rgb.png, depth_planar_cm.f32, world_xyz_cm.f64
│       └── MotionLog.csv                 # full trajectory of the trial
├── GroundTruthDrone/                     # real-drone targets
│   ├── gerolfing.json                    # outdoor target captures (GPS, attitude, image names)
│   ├── targets/                          # target images, e.g. 04-58-03-551-visible.jpg
│   └── <place>/GT<k>_<ts>.png + motion log   # e.g. indoor/, table/
├── LogDataDrone/<weather>/<run>/         # real-drone runs: log, dataLog, images/imageStart/End
└── motionDrone/<weather>/<run>/          # real-drone motion log, optional rgb.jpg (final snapshot)
```

- **Simulation:** each `logData<set>` run is paired with the `Final<n>` snapshot taken right after
  it, matched by timestamp. Its target is the GT named in its config, or the GT with the same number
  `GT<k>` if the GT folder was re-captured under a new timestamp (the camera pose must be identical).
- **Drone:** a run's target is found in this order:
  1. a run named `idx<k>` uses entry `k` of `gerolfing.json` (target image and target pose);
  2. otherwise the folder `GroundTruthDrone/<run name without trailing digits>/` (`indoor1` →
     `indoor/`, `table` → `table/`);
  3. otherwise the target image named in the run's log, searched by file name.

  A run without a ground-truth image (`clear/ground`) gets no image comparison; its algorithm plot is
  still made.

### Older run layout (e.g. `data/motionLog5_03/`)

```
data/motionLog5_03/
├── GroundTruth/GT<k>_.../                     # as above
└── <weather>/                                 # e.g. sunny
    ├── finalMotion/Final<n>_<YYYYmmdd_HHMMSS>/
    ├── images/imageStart<ts>.png, imageEnd<ts>.png
    └── matches/matches_<ts>.npz
```

This layout has no logs or configs. Runs that stopped before the 180 s timeout are treated as
settled, and the settle threshold is assumed (`--settle-px`, default 1.5 px).

---

## Running the evaluation

Run from this folder (`controls/`). There are three modes:

| what you want to redo | command | time on CPU |
|---|---|---|
| **Everything**: pose analysis, statistics, image comparison and all figures | `python analyze_final_motion.py` | about 1–2 h |
| **Only the plots** (and statistics), without the image models | `python analyze_final_motion.py --mode plots` | a few minutes |
| **Only the image comparison** (scores, pixel error, image panels) | `python analyze_final_motion.py --mode images` | about 1 h |

- **Image scores are cached** in `finalMotion/analysis/image_scores.json`. `--mode plots` reuses
  them, so the figures and `results.json` still contain the scores without loading DINOv2 or
  LightGlue. A pair that was already scored is never scored again; delete `image_scores.json` to
  force a full re-scoring.
- **`--mode images`** skips the summary figures, the per-trial translation/rotation figures and the
  drone algorithm and pose plots, but still writes complete `results.json`, `trials.csv` and
  `summary.*` (the pose numbers are recomputed; they take seconds).

Further options:

```powershell
# only some configurations (the <set> suffixes of the logData/motion folders), no drone runs
python analyze_final_motion.py --sets 5_05 5_05fog 5_05ess --no-drone

# skip the ~700 per-trial simulation figures
python analyze_final_motion.py --no-trial-plots

# older run layout
python analyze_final_motion.py --run data/motionLog5_03 --out data/motionLog5_03/analysis1
```

| option | meaning |
|---|---|
| `--mode {all,plots,images}` | what to run (see above); default `all` |
| `--root` | data folder (default `finalMotion`) |
| `--sets` | simulation configurations to evaluate (default: every `logData*` folder found) |
| `--no-drone` | skip the real-drone runs |
| `--no-trial-plots` | skip the per-trial simulation figures |
| `--out` | output folder (default `<root>/analysis`) |
| `--run` | evaluate a folder in the older layout instead |
| `--settle-px` | `--run` layout only: assumed settle threshold in px |

On Windows, set `$env:PYTHONIOENCODING = "utf-8"` if the console fails to print the summary table.

### Outputs (`finalMotion/analysis/`)

| file | content |
|---|---|
| `results.json` | **Everything in one file:** metric definitions, GT poses, per-configuration statistics (all trials and settled only), per-trial results (config, file paths, image scores, error-over-time curve, per-frame rotation-estimate tracking) and the drone runs |
| `trials.csv` | one row per simulation trial with all metrics |
| `drone_runs.csv` | one row per drone run: outcome, image scores and pixel error, algorithm and logged-pose errors |
| `summary.md` | per-configuration table: mean ± SD (RMSE) of the pose errors, worst rotation error, time to stop, image score and pixel error from start to end |
| `summary.csv` | the same statistics in long format |
| `image_scores.json` | cache of the image comparisons (see above) |
| `figures/` | summary figures (see below) |
| `trials/<set>/` | per simulation trial: `GT<k>_<ts>_images.png` (target, Start and End with image score and pixel error), `_translation.png` and `_rotation.png` (trajectory vs target) |
| `drone/<weather>/` | per drone run: `<run>_images.png`, `_algorithm.png` (the algorithm's per-frame image error, rotation estimate and commands), `_camera_rotation.png`, `_drone_rotation.png`, `_translation.png` (GPS, where available) |

Summary figures in `figures/`:

| figure | shows |
|---|---|
| `compensation_base.png` | actual vs rotation-compensating translation, and the per-trial lateral error budget (base set) |
| `errors_per_set.png` | final rotation, translation, effective lateral and forward error per trial, for every configuration |
| `rotation_estimators.png` | PNEC vs NEC vs essential matrix: per-frame rotation estimate against the truth |
| `error_over_time_<set>.png` | true translation and rotation error over time for every trial of a configuration |
| `convergence_basin.png` | start offset and start field-of-view overlap vs outcome |

---

## What is measured

All pose errors are computed from the **true** camera poses exported by the simulator, never from the
algorithm's own estimates. Every trial is included with the pose at the moment the system stopped,
either by meeting the stopping criterion or at the 180 s timeout.

| metric | meaning |
|---|---|
| `alpha_deg` | final rotation error, deg |
| `t_cm` | final translation error, cm (GT camera frame: x right, y down, z forward) |
| `t_star_cm` | translation that best compensates the residual rotation in the image (≈ α · scene depth), fitted with the GT depth at the matched keypoints |
| `eps_lat_cm` | effective (visible) lateral error = lateral part of t − t\*. This is the image-relevant position error. |
| `eps_fwd_cm` | error along the optical axis (weakly observable with a single camera) |
| `deadband_cm` | lateral offset that the pixel settle threshold allows at the scene depth |
| `rot_est_err_*` | per-frame accuracy of the algorithm's rotation estimate against the truth |
| `start_overlap` | fraction of the target view visible from the start pose |
| `img_start`, `img_end`, `img_final` | image score in [0, 1] of the Start / End / final-snapshot image against the target (`repositioning_score.py`: 0.2 · DINOv2 global similarity + 0.4 · overlap + 0.4 · alignment) |
| `img_<x>_global`, `_overlap`, `_alignment` | the three components of the image score |
| `img_start_px`, `img_end_px`, `img_final_px` | **pixel error**: median distance, in pixels of the 1920×1080 frame, between the matched points of the target and the image (SuperPoint/LightGlue matches that survive RANSAC). All targets and frames are 1920×1080, so this is directly comparable across simulation and drone runs. |

Notes on the image comparison:

- The pixel error measures how far the view still is from the target view, in the same unit as the
  controller's stopping threshold. It contains both rotation and translation misalignment.
- Drone images carry the camera's burned-in on-screen overlay (info box, "Live Stream" box, logo).
  Feature matches inside these regions are ignored, otherwise the overlay would match itself with
  zero displacement.
- A truncated image (incomplete copy off the drone) is compared on the rows it has, with the target
  cropped to the same rows.

Why the translation error never reaches zero, and why `eps_lat_cm` is the meaningful position
error, is explained in `finalMotion/RESULTS.md` §3.

---

## Status

- Simulation results: complete (see the RESULTS documents).
- Real-drone results: evaluated by image comparison (target vs Start/End), the algorithm's own
  per-frame estimates and the logged gimbal attitude. GPS positions are logged but cannot measure
  the true position error without RTK.
