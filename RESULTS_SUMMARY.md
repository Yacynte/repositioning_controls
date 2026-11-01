# Results Summary: Monocular Return-to-Location Repositioning

The drone returns to a reference pose using only its monocular camera and a stored target image.
The rotation is estimated with PNEC, and the drone is driven by a tanh control law. The method was
evaluated in simulation (Unreal Engine, with exact ground-truth poses and depth) and on a real drone.

Every statistic uses the pose at the moment the system stopped: either when the stopping criterion
was met ("settled") or at the timeout. The timeout is a deliberate stop (180 s in simulation, well
inside the 10-minute repositioning budget), not a failure. Values are **mean ± SD (RMSE)** over trials.

> **To fill in:** the values marked ⟨…⟩ come from the image comparison run on the GPU machine
> (`finalMotion/analysis/summary.md` and `drone_runs.csv`).

---

## 1. Simulation setup

| | value |
|---|---|
| Target positions | GT1–GT6, 3 random starts each → 18 trials per configuration (234 in total) |
| Start offset | 1.9–3.4 m translation, 14–28° rotation, 49–71 % field-of-view overlap with the target |
| Stopping criterion | image translation error < 1.5 px and estimated rotation < 0.1°, held for 2 s |
| Camera | 1920 × 1080, 90° horizontal FOV |
| Scene depth (matched keypoints) | 140–230 m |
| Conditions | clear, light rain, rain, light snow, heavy snow, fog |
| Ablations (clear weather) | settle threshold 1.0 / 0.5 px; linear / PD control; NEC / essential matrix; parallax-based translation |

**Error metrics.** α is the true final rotation error and |t| the true final translation error. At
these scene depths a small residual rotation α is best compensated in the image by a lateral
translation t\* ≈ α·Z (Z = scene depth). The controller aligns *images*, so it converges to t\*,
not to t = 0. The real, visible misalignment is the **effective error ε = t − t\***. Its lateral
part ε_lat is what the image can observe; the forward part ε_fwd is only weakly observable with one
camera.

## 2. Simulation results

### 2.1 Clear weather (base configuration)

- **18/18** trials met the stopping criterion, after 30.8 s on average (median 27 s).
- Final rotation error **α = 0.147 ± 0.086° (RMSE 0.169°)**, from 14–28° at the start.
- Final translation error |t| = 73.6 ± 24.1 cm. About 39 cm of it is the translation that
  compensates the residual rotation (Z ≈ 192 m).
- **Effective lateral error ε_lat = 39.7 ± 16.5 cm (RMSE 42.8 cm) ≈ 2 px**, of the order of the 1.5 px
  stopping threshold (30 cm at this depth).
- A separate clear-weather run (5_03, 6 trials) confirms this: 6/6 settled,
  α = 0.158 ± 0.103°, ε_lat = 49.1 ± 25.1 cm.

### 2.2 Robustness to weather

| condition | stopped by criterion | α [deg] | α max [deg] | ε_lat [cm] | time to stop, median [s] |
|---|---|---|---|---|---|
| clear | 18/18 | 0.15 ± 0.09 | 0.41 | 39.7 ± 16.5 | 27 |
| light rain | 15/18 | 0.21 ± 0.13 | 0.50 | 29.8 ± 14.9 | 30 |
| rain | 13/18 | 0.17 ± 0.11 | 0.38 | 29.3 ± 12.4 | 48 |
| light snow | 10/18 | 0.25 ± 0.14 | 0.52 | 31.6 ± 16.4 | 138 |
| heavy snow | 8/18 | 0.32 ± 0.21 | 0.78 | 29.7 ± 17.0 | 180 |
| fog | 15/18 | 0.51 ± 0.23 | 0.93 | 28.1 ± 17.9 | 33 |

- **The method converges in every condition.** Every trial ended with α < 1°.
- **Weather lowers the rotation precision:** about 0.15–0.2° in clear weather and rain, 0.25–0.3° in
  snow, and about 0.5° in fog (within 1° in every trial). The cause is fewer and noisier feature
  matches: about 114 initial matches in clear weather vs 67–73 in fog and heavy snow.
- **The image is still aligned in bad weather.** ε_lat stays at 28–32 cm in every condition; the
  larger rotation error is compensated by a larger translation.
- **Bad weather mainly costs time.** The strict criterion is met less often (heavy snow 8/18), and
  those runs stop at the 180 s timeout while holding a pose close to the target (median |t| 1.1 m,
  down from 2.6 m; α 0.04–0.81°).

### 2.3 Rotation estimator

| estimator | stopped by criterion | final α [deg] | α max [deg] | trials with α > 1° | per-frame estimate error (median) |
|---|---|---|---|---|---|
| **PNEC** | **18/18** | **0.15 ± 0.09** | 0.41 | 0 | **0.24°** |
| NEC | 16/18 | 0.36 ± 0.12 | 0.57 | 0 | 0.36° |
| essential matrix | 1/18 | 1.05 ± 1.32 | 4.38 | 7 | 0.40° |

- Weighting each correspondence by its keypoint covariance (NEC → PNEC) reduces the final rotation
  error by a factor of 2.4.
- The essential matrix becomes ill-conditioned as the baseline goes to zero near the target. Its
  estimate then jumps by about 0.36° from frame to frame, 3× more than PNEC's. As a result it rarely
  meets the stopping criterion and ends above 1° in 7 of 18 trials.

### 2.4 Other ablations

- **Settle threshold.** 0.5 px instead of 1.5 px reduces ε_lat by about 35 % (39.7 → 26.0 cm), at the
  cost of a longer time to stop (median 27 → 46 s). 1.5 px is a good trade-off.
- **Control law.** Linear and PD control settle as reliably as tanh (18/18) but leave larger
  rotation errors (0.21° vs 0.15°). tanh gives the most precise final pose.
- **Translation source.** The parallax-based translation signal settles in only 3/18 trials, and its
  forward error scatters twice as much as with the flow-based signal.

## 3. Image similarity

The final view was compared with the target image using a two-stage score:
DINOv2 global appearance (weight 0.3) and SuperPoint + LightGlue geometric alignment (weight 0.7),
after removing wrong matches with RANSAC. In addition, the **pixel error** is the median distance, in
pixels of the 1920 × 1080 frame, between matched target and image points. This makes it directly
comparable between the simulation and the real drone, and with the controller's own stopping
threshold.

| condition | image score start → end | pixel error start → end [px] |
|---|---|---|
| clear | ⟨…⟩ → ⟨…⟩ | ⟨…⟩ → ⟨…⟩ |
| light rain | ⟨…⟩ → ⟨…⟩ | ⟨…⟩ → ⟨…⟩ |
| rain | ⟨…⟩ → ⟨…⟩ | ⟨…⟩ → ⟨…⟩ |
| light snow | ⟨…⟩ → ⟨…⟩ | ⟨…⟩ → ⟨…⟩ |
| heavy snow | ⟨…⟩ → ⟨…⟩ | ⟨…⟩ → ⟨…⟩ |
| fog | ⟨…⟩ → ⟨…⟩ | ⟨…⟩ → ⟨…⟩ |

⟨Interpretation to add once the numbers are in.⟩

## 4. Real-drone results

The drone has no RTK, so GPS cannot measure the true position error. Returning to the same GPS
coordinates does not bring the camera back to the same view, and that is the problem the image-based
correction addresses. The runs are therefore evaluated in the image: by the algorithm's own pixel
error, by the camera rotation relative to the target attitude, and by the image comparison with the
target image.

| run | target | stop | duration [s] | algorithm pixel error start → end [px] | camera yaw error start → end [deg] | GPS distance from target start → end [m] | image pixel error start → end [px] |
|---|---|---|---|---|---|---|---|
| outdoor, clear, idx1 | waypoint 1 | timeout (90 s) | ~90 | 15.1 → **4.3** | 17.4 → 2.7 | 0.03 → 3.76 | ⟨…⟩ |
| outdoor, clear, idx2 | waypoint 2 | timeout (90 s) | ~90 | 28.5 → 21.7 | 10.3 → **0.0** | 0.05 → 3.70 | ⟨…⟩ |
| outdoor, rain, idx4 | waypoint 4 | timeout | 296 | 4.7 → 12.3 | 2.9 → 1.7 | 0.07 → 5.19 | ⟨…⟩ |
| lab, table | table target | **settled** | 8 | — | −10.3 → −1.1 | — (no GPS) | ⟨…⟩ |
| lab, indoor1 | indoor target | **settled** | 14 | — | 21.3 → −2.0 | — (no GPS) | ⟨…⟩ |
| lab, indoor2 | indoor target | **settled** | 49 | — | −20.1 → −1.3 | — (no GPS) | ⟨…⟩ |

The algorithm pixel error is the image error the controller computes itself (median over the first and
last 5 % of frames). Camera yaw is the gimbal yaw relative to the attitude recorded with the target
image. A run without a target image (clear/ground) is excluded.

- **In the lab, all three runs met the stopping criterion**, within 8–49 s, from 10–21° of camera yaw
  error to 1–2°.
- **Outdoors, GPS reported the drone at the target (3–7 cm) while the image was still misaligned**
  (15–29 px). The algorithm then moved the drone 3.7–5.2 m away from the GPS target position to
  align the image. This is the GPS-only repositioning error the method is designed to correct.
- **idx1** reduced its image error from 15 px to about 4 px and its camera yaw error from 17° to
  under 3° within the 90 s budget. **idx2** corrected its camera yaw completely (10° → 0°), but its
  image error decreased only from 29 to 22 px before the timeout.
- **In rain (idx4) the image error increased** from about 5 px to 12 px over the run. ⟨Check against
  the image comparison and the algorithm plot before interpreting: possible causes are rain on the lens,
  wind, or the target image's lighting.⟩
- The drone's own pitch and roll are disturbed by wind and controlled by the flight controller, not by
  the algorithm. The correction happens through the gimbal and the position commands.

## 5. Limitations

1. **Metric precision is bounded by scene depth.** At 140–230 m, 1 px corresponds to 15–24 cm
   laterally, and the forward error is only weakly observable with a single camera.
2. **The true rotation error at the stop (0.15°) exceeds the 0.1° stopping threshold**, because the
   criterion uses the *estimated* rotation, which itself has an error of about 0.18° near convergence.
3. **Only one strict stopping criterion was tested.** In bad weather it is often not met, and the
   run stops at the timeout while already close to the target. A condition-dependent criterion
   (e.g. 2–3 px after 60 s) could stop earlier at the same precision.
4. **Start overlap was 49–71 % only.** The lower limit of the convergence basin is not established.
5. **The real-drone tests are few** (3 outdoor runs, 3 lab runs), without RTK ground truth. They show
   the behaviour on hardware but do not give statistically meaningful accuracy figures.
