#!/usr/bin/env python3
"""
Is the final translation error a compensation of the rotation error?
Observability (Fisher-information) analysis with ground-truth depth.

Idea
----
Your matcher/controller only "sees" the pose through keypoint positions. Linearise the
projection of the GT scene points around the GT pose:

    du = J e,     e = [w_x, w_y, w_z (rad), t_x, t_y, t_z (m)]   (final pose vs GT, GT CV frame)

With keypoint noise sigma (px), the information matrix is  I = J^T J / sigma^2  and the
pose covariance is  Sigma = I^-1.  Then:

  * correlation(w_y, t_x), correlation(w_x, t_y) close to +-1  -> rotation and translation
    are coupled (bas-relief / rotation-translation ambiguity: t ~ -Z * w)
  * the smallest eigenvector of I is the coupled direction the image cannot resolve
  * Mahalanobis distance d^2 = e^T I e = |J e|^2 / sigma^2 of the ACTUAL final-pose error:
      d^2 < chi2_6(95%)  -> the final pose is statistically indistinguishable from the GT
                            with this many keypoints at this noise level
  * share of the (normalised) error lying along the weak eigenvectors: high -> the error
    is the ambiguity, not a random controller error
  * best compensating translation for the observed rotation (closed form):
      t* = -(J_t^T J_t)^-1 J_t^T J_w w     -> compare per axis, with sign, to the actual t

Usage
-----
  python pose_observability.py                               # every trial of RUN_DIR (data/motionLog5_03)
  python pose_observability.py --run data/motionLog5_03 [--results data/results5_03]
  python pose_observability.py --gt <GT snapshot dir> --final <Final snapshot dir>
         [--keypoints kps.npy] [--n-points 500] [--sigma-px 1.0] [--out dir]

  --keypoints : Nx2 matched keypoints in the GT image (OpenCV pixel convention). Strongly
                recommended: information grows with the number of points, so use what your
                matcher really had. Without it, --n-points random valid pixels are used.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

import rt_ambiguity_analysis as rt

RUN_DIR = "data/motionLog5_03"

# Matcher output per weather: <run>/<weather>/matches/matches_<timestamp>.npz.
# mkpts1 are the keypoints in the target (GT) image - those are back-projected
# with the GT depth. mkpts0 are the same matches in the live camera image.
MATCHES_SUBDIR = "matches"
MATCH_KEY = "mkpts1"
OBSERVED_KEY = "mkpts0"

NAMES = ["w_x (pitch, rad)", "w_y (yaw, rad)", "w_z (roll, rad)", "t_x (right, m)", "t_y (down, m)", "t_z (fwd, m)"]
SHORT = ["w_x", "w_y", "w_z", "t_x", "t_y", "t_z"]


def pose_from_params(e):
    return Rotation.from_rotvec(e[:3]).as_matrix(), e[3:]


def jacobian(P, K, eps_rot=1e-6, eps_t=1e-4):
    """d(pixel)/d(e) at e = 0 by central differences (2N x 6), same projection as rt.project."""
    N = len(P)
    J = np.empty((2 * N, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps_rot if k < 3 else eps_t
        up = rt.project(P, *pose_from_params(d), K)
        dn = rt.project(P, *pose_from_params(-d), K)
        J[:, k] = ((up - dn) / (2 * d[k])).ravel()
    return J


def load_keypoints(path, key=MATCH_KEY):
    """
    Nx2 GT-image keypoints from a .npy, or from a matcher .npz (matches_onnx.py):
    mkpts1 = target (GT) image points, mkpts0 = live camera points.
    """
    if Path(path).suffix == ".npz":
        with np.load(path) as z:
            return np.asarray(z[key], np.float64)
    return np.load(path)


def find_match_files(matches_dir):
    """Sorted [(datetime, path)] of matches_<YYYY_mm_dd_HH_MM_SS>.npz files."""
    found = []
    for path in Path(matches_dir).glob("matches_*.npz"):
        m = re.match(r"^matches_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.npz$", path.name)
        if m is None:
            print(f"WARNING: Could not parse match file: {path}")
            continue
        found.append((datetime.strptime(m.group(1), "%Y_%m_%d_%H_%M_%S"), path))
    return sorted(found)


def match_files_to_finals(finals, match_files):
    """
    {final_index: path} - each Final gets the latest unused match file written at or
    before it (the matches are saved ~10 s before the Final snapshot).
    """
    matched, used = {}, set()
    for n in sorted(finals, key=lambda i: finals[i][0]):
        candidates = [k for k, (t, _) in enumerate(match_files)
                      if t <= finals[n][0] and k not in used]
        if candidates:
            used.add(candidates[-1])
            matched[n] = match_files[candidates[-1]][1]
    for k, (_, path) in enumerate(match_files):
        if k not in used:
            print(f"WARNING: match file not matched to any Final: {path.name}")
    return matched


def analyze(gt_dir, final_dir, keypoints=None, n_points=500, sigma_px=1.0, seed=0, verbose=True,
            observed=None):
    """
    keypoints: Nx2 GT-image points (array, .npy or matcher .npz), or None for random pixels.
    observed:  optional Nx2 positions of the same matches in the live image (mkpts0),
               to compare the measured offset with the one the pose error predicts.
    """
    log = print if verbose else (lambda *a, **k: None)
    gt, fin = rt.load_meta(gt_dir), rt.load_meta(final_dir)
    depth = rt.load_depth_m(gt_dir, gt)
    K = rt.intrinsics(gt)

    kps = load_keypoints(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
    n_keypoints = None if kps is None else len(kps)
    if kps is not None:
        # same validity test as rt.ref_points (sky / no depth is dropped)
        H, W = depth.shape
        ui = np.clip(np.round(kps[:, 0]).astype(int), 0, W - 1)
        vi = np.clip(np.round(kps[:, 1]).astype(int), 0, H - 1)
        Zk = depth[vi, ui]
        valid = np.isfinite(Zk) & (Zk > 0)
        if observed is not None:
            observed = np.asarray(observed, np.float64)[valid] + 0.5  # OpenCV -> pixel-centre coords
    P, uv, _ = rt.ref_points(depth, K, kps)
    if kps is None and len(P) > n_points:
        sel = np.random.default_rng(seed).choice(len(P), n_points, replace=False)
        P, uv = P[sel], uv[sel]
    N = len(P)

    # actual error of the final pose (GT camera CV frame)
    R, t = rt.relative_pose(gt, fin)
    w = Rotation.from_matrix(R).as_rotvec()
    e = np.concatenate([w, t])

    # information, covariance, correlation
    J = jacobian(P, K)
    H = J.T @ J                       # = sigma^2 * information
    Sigma = sigma_px ** 2 * np.linalg.inv(H)
    std = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(std, std)

    # eigen-analysis in a unit-free (whitened) parameterisation
    s = np.sqrt(np.diag(H))           # px per unit of each parameter
    Hn = H / np.outer(s, s)
    evals, evecs = np.linalg.eigh(Hn)  # ascending: evecs[:,0] = least observable direction
    en = e * s                        # error in the same normalised coordinates
    share = (evecs.T @ en) ** 2 / max(en @ en, 1e-30)

    # Mahalanobis distance of the actual error and the residual it leaves in the image
    d2 = float(e @ H @ e) / sigma_px ** 2
    d2_lim = float(chi2.ppf(0.95, 6))
    resid_lin = J @ e
    resid_exact = (rt.project(P, R, t, K) - uv).ravel()
    rms = lambda r: float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2))

    # best translation for the observed rotation, and the image error it would leave
    Jw, Jt = J[:, :3], J[:, 3:]
    t_star = -np.linalg.solve(Jt.T @ Jt, Jt.T @ (Jw @ w))
    rot_only = rms(Jw @ w)
    with_actual_t = rms(resid_lin)
    with_t_star = rms(Jw @ w + Jt @ t_star)

    Z = P[:, 2]
    Z_h = 1.0 / np.mean(1.0 / Z)

    # ---------------------------------------------------------------- report
    log(f"\nPoints: {N} ({'matched keypoints' if kps is not None else 'random valid pixels'}),"
        f" keypoint noise sigma = {sigma_px} px, scene depth harmonic mean {Z_h:.1f} m")
    log("\n--- Actual final-pose error (GT camera frame: x right, y down, z forward)")
    for k in range(6):
        val = np.degrees(e[k]) if k < 3 else e[k]
        sd = np.degrees(std[k]) if k < 3 else std[k]
        unit = "deg" if k < 3 else "m"
        log(f"  {SHORT[k]}: {val:+9.4f} {unit}    1-sigma observability {sd:9.4f} {unit}   ({abs(val) / sd:5.2f} sigma)")

    log("\n--- Coupling (correlation of the pose estimate from these keypoints)")
    log(f"  corr(w_y, t_x) = {corr[1, 3]:+.4f}   (yaw   <-> sideways)")
    log(f"  corr(w_x, t_y) = {corr[0, 4]:+.4f}   (pitch <-> up/down)")
    log(f"  corr(w_z, t_z) = {corr[2, 5]:+.4f}   (roll  <-> forward, normally ~0)")

    log("\n--- Least observable directions (eigenvectors of the normalised information)")
    for i in range(2):
        v = evecs[:, i] / np.max(np.abs(evecs[:, i]))   # px-normalised units: w and t comparable
        comp = ", ".join(f"{SHORT[k]} {v[k]:+.2f}" for k in range(6) if abs(v[k]) > 0.05)
        log(f"  #{i + 1}: eigenvalue {evals[i]:.2e}   [{comp}]   share of your error: {share[i]:.1%}")
    log(f"  -> {share[:2].sum():.1%} of the (normalised) final-pose error lies in the 2 weakest directions")

    log("\n--- Compensation test (image RMS residual of the GT keypoints)")
    log(f"  rotation error alone          {rot_only:8.3f} px")
    log(f"  rotation + actual translation {with_actual_t:8.3f} px   (exact projection: {rms(resid_exact):.3f} px)")
    log(f"  rotation + best translation t*{with_t_star:8.3f} px")
    log(f"  t* = [{t_star[0]:+.3f}, {t_star[1]:+.3f}, {t_star[2]:+.3f}] m    actual t = [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m")
    log(f"  rule of thumb t ~ -Z*w:  -Z_h*w_y = {-Z_h * w[1]:+.3f} m (vs t_x), Z_h*w_x = {Z_h * w[0]:+.3f} m (vs t_y)")

    log("\n--- Statistical verdict")
    log(f"  Mahalanobis d^2 = {d2:.2f}   (chi2_6 95% limit {d2_lim:.2f})")
    if d2 < d2_lim:
        log("  -> the final pose is INDISTINGUISHABLE from the GT with these keypoints at this noise:")
        log("     the remaining rotation/translation error is unobservable, not a controller fault.")
    else:
        log("  -> the final pose is distinguishable from the GT: the controller could have done better.")
    if with_actual_t < 0.5 * rot_only:
        log("  -> the translation error cancels most of the rotation's image shift (compensation).")

    rms_px = {"rotation_only": rot_only, "rotation_plus_translation": with_actual_t,
              "rotation_plus_t_star": with_t_star, "exact": rms(resid_exact)}
    measured = None
    if observed is not None:
        # offset the matcher actually saw (live - GT image), vs the one the pose predicts
        d_meas = observed - uv
        d_pred = rt.project(P, R, t, K) - uv
        rms_px["measured_matches"] = rms(d_meas)
        measured = {
            "rms_px": rms(d_meas),
            "median_px": float(np.median(np.linalg.norm(d_meas, axis=1))),
            "mean_offset_px": d_meas.mean(axis=0).tolist(),
            "predicted_mean_offset_px": d_pred.mean(axis=0).tolist(),
            "rms_measured_minus_predicted_px": rms(d_meas - d_pred),
        }
        log("\n--- Matcher keypoints at convergence (mkpts0 - mkpts1)")
        log(f"  measured RMS offset {measured['rms_px']:.3f} px   (pose predicts {rms(resid_exact):.3f} px)")
        log(f"  mean offset measured [{d_meas.mean(0)[0]:+.2f}, {d_meas.mean(0)[1]:+.2f}] px, "
            f"predicted [{d_pred.mean(0)[0]:+.2f}, {d_pred.mean(0)[1]:+.2f}] px")

    return {
        "keypoints_total": n_keypoints, "points": N,
        "point_source": "matched keypoints (mkpts1)" if kps is not None else "random valid pixels",
        "sigma_px": sigma_px, "depth_harmonic_mean_m": float(Z_h),
        "error": {SHORT[k]: float(np.degrees(e[k]) if k < 3 else e[k]) for k in range(6)},
        "observability_1sigma": {SHORT[k]: float(np.degrees(std[k]) if k < 3 else std[k]) for k in range(6)},
        "corr_wy_tx": float(corr[1, 3]), "corr_wx_ty": float(corr[0, 4]), "corr_wz_tz": float(corr[2, 5]),
        "error_share_weakest_2": float(share[:2].sum()),
        "rms_px": rms_px,
        "matcher_offset": measured,
        "t_star_m": t_star.tolist(), "t_m": t.tolist(),
        "mahalanobis_d2": d2, "chi2_95": d2_lim, "indistinguishable": bool(d2 < d2_lim),
    }


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def analyze_run(run_dir, results_dir, n_points=500, sigma_px=1.0, verbose=False):
    """
    Analyse every trial of a run folder laid out like data/motionLog5_03:
        <run>/GroundTruth/GT<id>_<timestamp>/            (meta.json, depth_planar_cm.f32, ...)
        <run>/<weather>/finalMotion/Final<n>_<timestamp>/ (meta.json, ...)
    Final<n> is compared with GT ((n - 1) % number_of_GTs) + 1, as in evaluate_repositioning.py.
    Writes <results_dir>/observability_results.yaml.
    """
    import yaml
    from evaluate_repositioning import find_snapshots, GT_SUBDIR, FINAL_SUBDIR

    run_dir, results_dir = Path(run_dir), Path(results_dir)
    gt_snapshots = find_snapshots(run_dir / GT_SUBDIR, "GT")
    if not gt_snapshots:
        raise RuntimeError(f"No GT<id>_<timestamp>/ snapshots in {run_dir / GT_SUBDIR}")
    gt_ids = sorted(gt_snapshots)

    weathers = sorted(p.name for p in run_dir.iterdir()
                      if p.is_dir() and p.name != GT_SUBDIR and (p / FINAL_SUBDIR).is_dir())
    print(f"{len(gt_ids)} GT snapshot(s), weather condition(s): {weathers}")

    trials = []
    for weather in weathers:
        finals = find_snapshots(run_dir / weather / FINAL_SUBDIR, "Final")
        match_files = match_files_to_finals(
            finals, find_match_files(run_dir / weather / MATCHES_SUBDIR))
        print(f"[{weather}] {len(finals)} Final snapshot(s), {len(match_files)} with a match file")
        for n in sorted(finals):
            final_dir = finals[n][1]
            gt_id = gt_ids[(n - 1) % len(gt_ids)]
            gt_dir = gt_snapshots[gt_id][1]

            match_file = match_files.get(n)
            if match_file is None:
                print(f"WARNING: no match file for Final{n}, using {n_points} random pixels")
                kps = observed = None
            else:
                with np.load(match_file) as z:
                    kps = np.asarray(z[MATCH_KEY], np.float64)
                    observed = np.asarray(z[OBSERVED_KEY], np.float64)

            res = analyze(gt_dir, final_dir, kps, n_points, sigma_px, verbose=verbose,
                          observed=observed)
            res["match_file"] = str(match_file) if match_file else None
            rms = res["rms_px"]
            print(f"{weather} | Final{n} -> GT{gt_id}  [{res['points']} pts]:  d^2 {res['mahalanobis_d2']:7.2f} "
                  f"({'indistinguishable' if res['indistinguishable'] else 'distinguishable'})   "
                  f"corr(w_y,t_x) {res['corr_wy_tx']:+.3f}  corr(w_x,t_y) {res['corr_wx_ty']:+.3f}   "
                  f"rms px: rot {rms['rotation_only']:.2f} -> rot+t {rms['rotation_plus_translation']:.2f} "
                  f"(best {rms['rotation_plus_t_star']:.2f})"
                  + (f"   measured {rms['measured_matches']:.2f}" if "measured_matches" in rms else ""))

            trials.append({"weather": weather, "trial": n, "gt_id": gt_id,
                           "gt_dir": str(gt_dir), "final_dir": str(final_dir), **res})

    def summarize(rows):
        return {
            "number_of_trials": len(rows),
            "trials_indistinguishable_from_gt": sum(r["indistinguishable"] for r in rows),
            "trials_translation_cancels_half_of_rotation_shift": sum(
                r["rms_px"]["rotation_plus_translation"] < 0.5 * r["rms_px"]["rotation_only"]
                for r in rows),
            "mean_mahalanobis_d2": mean([r["mahalanobis_d2"] for r in rows]),
            "mean_corr_wy_tx": mean([r["corr_wy_tx"] for r in rows]),
            "mean_corr_wx_ty": mean([r["corr_wx_ty"] for r in rows]),
            "mean_error_share_weakest_2": mean([r["error_share_weakest_2"] for r in rows]),
            "mean_rms_px": {k: mean([r["rms_px"].get(k) for r in rows])
                            for k in ("rotation_only", "rotation_plus_translation",
                                      "rotation_plus_t_star", "exact", "measured_matches")},
        }

    output = {
        "run_dir": str(run_dir),
        "points_per_trial": f"matched keypoints ({MATCH_KEY}), else {n_points} random pixels",
        "sigma_px": sigma_px,
        "chi2_6_95": float(chi2.ppf(0.95, 6)),
        "summary": {
            "overall": summarize(trials),
            "by_weather": {w: summarize([r for r in trials if r["weather"] == w]) for w in weathers},
        },
        "trials": trials,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "observability_results.yaml"
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)
    print(f"\nResults written to {out_file}")
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN_DIR,
                    help="run folder with GroundTruth/ and <weather>/finalMotion/ (all trials)")
    ap.add_argument("--results", default=None,
                    help="output folder for the run (default: motionLogX_Y -> resultsX_Y)")
    ap.add_argument("--gt", default=None, help="single GT snapshot dir (instead of --run)")
    ap.add_argument("--final", default=None, help="single Final snapshot dir (instead of --run)")
    ap.add_argument("--keypoints", default=None, help="single mode: .npy or matcher .npz (uses mkpts1)")
    ap.add_argument("--n-points", type=int, default=500)
    ap.add_argument("--sigma-px", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="single mode: optional folder for observability.json")
    ap.add_argument("--verbose", action="store_true", help="run mode: full report per trial")
    a = ap.parse_args()

    if a.gt or a.final:
        if not (a.gt and a.final):
            ap.error("--gt and --final must be given together")
        res = analyze(a.gt, a.final, a.keypoints, a.n_points, a.sigma_px)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / "observability.json").write_text(json.dumps(res, indent=2))
        return

    run = Path(a.run)
    results = Path(a.results) if a.results else run.with_name(run.name.replace("motionLog", "results"))
    analyze_run(run, results, a.n_points, a.sigma_px, a.verbose)


if __name__ == "__main__":
    main()
