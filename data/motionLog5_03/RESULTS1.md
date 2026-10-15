# Simulation Results — Run 5_03: Clear Weather, PNEC

> **Scope:** 6 repositioning trials in clear weather with the PNEC rotation estimator. There is one
> trial per target position, GT1–GT6. The run was recorded on **2026-10-08**, the same day and under
> the same scene lighting as the GT reference images. That makes it the only run in which the CLIP
> image comparison is a valid accuracy measure (see [§5](#5-image-similarity-clip)).
>
> All numbers come from `analyze_final_motion.py --run data/motionLog5_03` (outputs in
> `analysis/`: `results.json`, `trials.csv`, `summary.md`, `figures/`, `trials/`). Values are
> **mean ± SD (RMSE)** over the 6 trials unless stated otherwise. The metric definitions are the same
> as in the main results (`data/finalMotion/RESULTS.md`, §2.3). As there, every statistic uses the
> pose at the moment the system stopped. In this run all 6 trials stopped by the criterion, so no
> timeout cases occur.

---

## 1. Summary

- **6/6 trials converged and stopped**, after 27–61 s (median 36 s). The run has no logs, so
  "settled" is inferred from every run stopping well before the 180 s timeout (§6).
- **Final rotation error: 0.158 ± 0.103° (RMSE 0.184°)**, down from 18–27° at the start.
- **Final translation error: |t| = 98.7 ± 55.1 cm**, down from 2.1–2.8 m at the start, with a
  median reduction of 56 %. Of the lateral part, **45.4 cm** is the translation that compensates the
  residual rotation in the image (t ≈ α·Z at Z ≈ 206 m). The correlation between the actual and the
  compensating lateral translation is r = 0.89.
- **Effective (visible) lateral error ε_lat: 49.1 ± 25.1 cm (RMSE 54.2 cm) ≈ 2.2 px**. The 1.5 px
  stopping threshold corresponds to 32 cm at this depth.
- **CLIP similarity of the End image to the GT image: 0.996 ± 0.002** (0.994–0.998), up from
  0.892 at the start. The final view is practically indistinguishable from the target view.

---

## 2. Setup

| | value |
|---|---|
| Configuration | PNEC rotation estimator, clear weather. No config file was logged for this run; the 1.5 px settle threshold of the base configuration is assumed. |
| Trials | 6 (one per target position GT1–GT6) |
| Start offset, translation | 2.1–2.8 m (mean 2.6 m) |
| Start offset, rotation | 18.4–26.8° |
| Start field-of-view overlap with the target view | 51–64 % |
| Scene depth at the matched keypoints (harmonic mean) | 146–295 m (mean 206 m) |
| Camera | 1920 × 1080, 90° horizontal FOV, f = 960 px |

Ground truth: each final camera pose comes from the Final snapshot `meta.json` and is checked
against the last MotionLog row. They agree to within 0.001 cm and 0.001°. The compensating
translation t\* is fitted on the 124–247 keypoints matched at the end of each run, using the GT
depth map.

---

## 3. Final accuracy

### 3.1 Per trial

| GT | time [s] | start offset [cm / deg] | start overlap | α [deg] | \|t\| [cm] | \|t_lat\| [cm] | \|t\*_lat\| [cm] | **ε_lat [cm]** | ε_lat [px] | ε_fwd [cm] | dead band [cm] | Z [m] | CLIP start → end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GT1 | 31.7 | 268 / 26.8 | 51 % | 0.084 | 43.2 | 31.6 | 23.1 | **31.8** | 1.83 | −29.0 | 26.1 | 167 | 0.833 → 0.996 |
| GT2 | 61.0 | 270 / 19.1 | 61 % | 0.362 | 165.4 | 154.0 | 92.0 | **62.5** | 2.99 | +60.9 | 31.4 | 201 | 0.924 → 0.995 |
| GT3 | 27.6 | 272 / 26.1 | 51 % | 0.150 | 123.2 | 98.6 | 42.0 | **58.0** | 2.83 | −78.3 | 30.8 | 197 | 0.886 → 0.994 |
| GT4 | 40.1 | 213 / 18.4 | 64 % | 0.125 | 20.5 | 20.2 | 26.3 | **6.2** | 0.41 | −2.2 | 22.8 | 146 | 0.875 → 0.998 |
| GT5 | 26.9 | 285 / 24.2 | 55 % | 0.144 | 124.3 | 105.9 | 51.5 | **71.4** | 2.99 | −63.0 | 35.8 | 229 | 0.952 → 0.995 |
| GT6 | 40.4 | 267 / 19.4 | 62 % | 0.086 | 115.3 | 98.8 | 37.5 | **64.8** | 2.11 | −57.1 | 46.0 | 295 | 0.880 → 0.998 |

### 3.2 Aggregate

| quantity | mean ± SD (RMSE) |
|---|---|
| rotation error α | **0.158 ± 0.103° (0.184°)** |
| translation error \|t\| | 98.7 ± 55.1 cm (110.7 cm) |
| compensating translation \|t\*\| | 45.5 cm |
| effective error \|ε\| = \|t − t\*\| | 69.3 ± 36.6 cm (76.9 cm) |
| **effective lateral error ε_lat** | **49.1 ± 25.1 cm (54.2 cm) ≈ 2.2 px** |
| effective forward error ε_fwd | −28.1 ± 51.3 cm (54.7 cm) |
| settle dead band (1.5 px at Z) | 32.2 cm |
| RMS image misalignment of the final pose | 3.0 px |
| time to stop | 37.9 s (median 35.9 s) |

---

## 4. Interpretation

### 4.1 Translation compensates the residual rotation

The controller aligns images, not poses. At a scene depth of about 200 m, a rotation error of
0.158° shifts the image as much as a lateral translation of α·Z ≈ 56 cm. The exact compensating
translation, fitted on the matched keypoints, is 45 cm. The actual and compensating lateral
translations are strongly correlated (r = 0.89). For GT2 and GT4 they point in almost the same
direction (within 4°).

- **GT2** is the clearest case. It ended with the largest rotation error (0.36°) and the largest
  translation error (1.65 m). 0.92 m of that translation is the compensation of the rotation, so
  only 62 cm lateral (3 px) remains visible.
- **GT4** shows the ideal outcome. With a 0.125° residual rotation, the translation (20 cm) is
  almost entirely compensation, and only **6 cm (0.4 px)** of effective lateral error remains.

### 4.2 The remaining error is at the stopping threshold

The effective lateral error averages 2.2 px, slightly above the 1.5 px stopping threshold
(32 cm at this depth). The stopping criterion uses the median flow of the matched keypoints after
the rotation is subtracted. The 2.2 px here is measured at the harmonic-mean depth, so the two are
of the same order but not identical measures.

### 4.3 The forward direction is the weakest

The largest single component of the error is along the optical axis: ε_fwd = −28 ± 51 cm. Five of
the six trials ended behind the target, by 2–78 cm. At about 200 m scene depth, a 50 cm forward
offset changes the image scale by only 0.25 %, about 2 px at the image edge and less towards the
centre. A monocular camera
observes this direction poorly, and the stopping threshold cannot resolve it.

### 4.4 Convergence over time

![error over time](analysis/figures/error_over_time_5_03_sunny.png)
*True translation and rotation error over time, all 6 trials.*

- **Rotation:** in every trial it falls roughly exponentially from 18–27° to about 0.1–0.2° within
  15–20 s.
- **Translation:** it falls more slowly, and in four trials it levels off at about 1.1–1.2 m. That
  plateau is the compensation of the residual rotation (§4.1), not a stall.
- **GT2** shows a slow drift after about 40 s. The rotation grows from 0.2° to 0.36° and the
  translation from 1.2 m to 1.65 m, while the two keep compensating each other in the image. The
  run stopped at 61 s. The drone moved along the rotation–translation ambiguity: the image hardly
  changed, but the pose drifted. This is the main weakness of a purely image-based stopping
  criterion, and the reason this trial took longest.

---

## 5. Image similarity (CLIP)

CLIP ViT-B/32 cosine similarity between the GT image and the Start/End camera images:

| GT | Start | End | gain | Final snapshot vs GT |
|---|---|---|---|---|
| GT1 | 0.833 | 0.996 | +0.163 | 0.996 |
| GT2 | 0.924 | 0.995 | +0.071 | 0.995 |
| GT3 | 0.886 | 0.994 | +0.108 | 0.995 |
| GT4 | 0.875 | 0.998 | +0.123 | 0.998 |
| GT5 | 0.952 | 0.995 | +0.043 | 0.996 |
| GT6 | 0.880 | 0.998 | +0.118 | 0.998 |
| **mean** | **0.892** | **0.996** | **+0.104** | **0.996** |

This run was recorded under the same lighting as the GT images, so the End-vs-GT similarity
reflects alignment only. Every trial ends at 0.994–0.998. The streamed End image and the
engine-rendered Final snapshot give the same score, so the video stream adds no measurable
degradation.

CLIP also shows a limit of image-level metrics. GT2 (|t| = 1.65 m, α = 0.36°) and GT4
(|t| = 0.21 m, α = 0.13°) score almost the same (0.995 vs 0.998). Image similarity confirms that the
view was recovered, but it cannot distinguish metric pose accuracy at this scene depth. The
depth-based metrics in §3 are needed for that.

> **Note for comparison with the finalMotion runs (2026-10-10):** the scene lighting changed between
> the two days. Against GT images re-captured under the new lighting, the finalMotion base set
> reaches an End-vs-GT score of 0.976 (0.954–0.992). The remaining gap to the 0.996 here comes from
> the dynamic sky, since that GT was captured three days after the runs.

Per-trial image panels (GT | Start | End) and the translation/rotation-vs-GT plots are in
`analysis/trials/5_03_sunny/`.

---

## 6. Limitations of this run

1. **Small sample:** 6 trials, one per position. The SDs and RMSEs are indicative only. The
   finalMotion base set (18 trials, 3 per position) is the statistically stronger result.
2. **No logs.** No config, log or data log was saved for this run. Therefore:
   - the outcome (settled or timed out) is inferred from the run durations (all ≤ 61 s, far below
     the 180 s timeout);
   - the settle threshold is assumed to be 1.5 px;
   - the per-frame rotation-estimate accuracy (finalMotion §6.2) and the initial match count cannot
     be computed.
3. **Version differences.** 5_03 predates the finalMotion runs. Whether the software and parameters
   were identical is not recorded, so the differences from the finalMotion base set should not be
   over-interpreted. Compared with that set, 5_03 has a similar α (0.158° vs 0.147°) and a somewhat
   larger ε_lat (49 vs 40 cm) and ε_fwd scatter.

---

## 7. Reproducing

```bash
# from controls/
python analyze_final_motion.py --run data/motionLog5_03            # with CLIP + per-trial figures
python analyze_final_motion.py --run data/motionLog5_03 --no-clip  # numbers only
```
