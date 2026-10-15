# Drone repositioning: controls and evaluation

Python side of a monocular **return-to-location** system for a drone. A target image is captured at a
reference pose. Later, the drone starts from a random pose near the target and flies back using only
its camera image and the stored target image.

The image matching, rotation estimation (PNEC) and control run in the main C++/Python pipeline. This
folder contains the drone/controller interface scripts and the **evaluation** of the simulation runs
in Unreal Engine.

Results:

- [data/finalMotion/RESULTS.md](data/finalMotion/RESULTS.md): main results. 13 configurations ×
  18 trials: clear weather, rain, snow, fog, and ablations of the settle threshold, control law,
  rotation estimator and translation source.
- [data/motionLog5_03/RESULTS1.md](data/motionLog5_03/RESULTS1.md): an earlier clear-weather PNEC
  run (6 trials).

---

## Repository contents

| file | purpose |
|---|---|
| **`analyze_final_motion.py`** | **Main evaluation script.** Computes all results, statistics and figures (see below). |
| `rt_ambiguity_analysis.py` | Geometry helpers used by the evaluation: camera poses from `meta.json`, depth loading, projection, rotation–translation ambiguity. |
| `clip_comparison.py` | Earlier stand-alone CLIP comparison for the old `groundTruths*/motionLog*` layout. Superseded by `analyze_final_motion.py`. |
| `evaluate_repositioning.py`, `pose_observability.py` | Earlier analysis attempts (compensation verdict, Fisher-information analysis). Kept for reference. |
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

`requirements.txt` installs CLIP from GitHub (`git+https://github.com/openai/CLIP.git`), so `git`
must be available. The first run downloads the CLIP ViT-B/32 weights (about 350 MB). A GPU is used
automatically if PyTorch finds one; on CPU the CLIP part takes most of the run time.

---

## Data

The raw simulation data is **not in the repository**: depth maps alone are about 60 MB per snapshot.
Only the results are versioned:

- `RESULTS*.md`, `results.json`, `summary.*`, `trials.csv` and the summary figures,
- the GT4 target image with its `meta.json`, and the GT4 per-trial figures as an example.

To reproduce the results, place the raw data under `data/` in the following layout.

### `data/finalMotion/` (main runs)

```
data/finalMotion/
├── GroundTruth/
│   └── GT<k>_<YYYYmmdd_HHMMSS>/        # one per target position, k = 1..6
│       ├── rgb.png                     # target image
│       ├── meta.json                   # exact camera pose + intrinsics
│       ├── depth_planar_cm.f32         # depth map (float32, HxW, cm)
│       ├── world_xyz_cm.f64            # world position per pixel (float64, HxWx3, cm)
│       └── MotionLog.csv               # drone pose at the target
├── logData<set>/                       # one per configuration, e.g. logData5_05, logData5_05fog
│   ├── config_<ts>.txt                 # ablation config, incl. target=GroundTruth/GT<k>_.../rgb.png
│   ├── log_<ts>.txt                    # controller log ("Maintained target position" or "Timed Out")
│   ├── dataLog_<ts>.csv                # per-frame estimated errors
│   ├── matches_<ts>.npz                # last feature matches
│   └── images/imageStart<ts>.png, imageEnd<ts>.png
└── motion<set>/                        # same <set> suffix as logData<set>
    └── Final<n>_<YYYYmmdd_HHMMSS>/     # snapshot at the end of each trial
        ├── meta.json                   # true final camera pose
        ├── rgb.png, depth_planar_cm.f32, world_xyz_cm.f64
        └── MotionLog.csv               # full trajectory of the trial
```

- Each `logData<set>` run is paired with the `Final<n>` snapshot taken right after it, matched by
  timestamp.
- Each run's target is the GT named in its config. If the GT folder has been re-captured under a new
  timestamp, the script matches it by its number `GT<k>` instead. The camera pose must then be
  identical.

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

Run from this folder (`controls/`):

```powershell
# all configurations in data/finalMotion: numbers, CLIP, summary figures and per-trial figures
python analyze_final_motion.py

# faster: skip CLIP and/or the ~700 per-trial figures (about 2 min instead of about 1 h on CPU)
python analyze_final_motion.py --no-clip --no-trial-plots

# only some configurations (the <set> suffixes of the logData/motion folders)
python analyze_final_motion.py --sets 5_05 5_05fog 5_05ess

# older run layout
python analyze_final_motion.py --run data/motionLog5_03 --out data/motionLog5_03/analysis1
```

| option | meaning |
|---|---|
| `--root` | data folder in the finalMotion layout (default `data/finalMotion`) |
| `--sets` | configurations to evaluate (default: every `logData*` folder found) |
| `--run` | evaluate a folder in the older layout instead |
| `--out` | output folder (default `<root>/analysis`) |
| `--no-clip` | skip the CLIP image similarity |
| `--no-trial-plots` | skip the per-trial figures |
| `--settle-px` | `--run` layout only: assumed settle threshold in px |

On Windows, set `$env:PYTHONIOENCODING = "utf-8"` if the console fails to print the summary table.

### Outputs (`<root>/analysis/`)

| file | content |
|---|---|
| `results.json` | **Everything in one file:** metric definitions, GT poses, per-configuration statistics (all trials and settled only), and per-trial results incl. config, file paths, CLIP scores, the error-over-time curve and the per-frame rotation-estimate tracking |
| `trials.csv` | one row per trial with all metrics |
| `summary.md` | per-configuration table (mean ± SD (RMSE), worst rotation error, time to stop, CLIP) |
| `summary.csv` | the same statistics in long format |
| `figures/` | summary figures: rotation–translation compensation, errors per configuration, convergence vs start offset/overlap, rotation-estimator comparison, error over time per configuration |
| `trials/<set>/` | per trial: `GT<k>_<ts>_images.png` (GT, Start and End images with CLIP scores), `_translation.png` and `_rotation.png` (trajectory vs target) |

---

## What is measured

All errors are computed from the **true** camera poses exported by the simulator, never from the
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
| `clip_start`, `clip_end` | CLIP ViT-B/32 cosine similarity of the Start/End image to the target image |
| `start_overlap` | fraction of the target view visible from the start pose |

Why the translation error never reaches zero, and why `eps_lat_cm` is the meaningful position
error, is explained in [data/finalMotion/RESULTS.md §3](data/finalMotion/RESULTS.md).

---

## Status

- Simulation results: complete (see the RESULTS documents).
- Real-drone results: to follow.
