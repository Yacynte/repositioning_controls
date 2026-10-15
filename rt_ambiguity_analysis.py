#!/usr/bin/env python3
"""
Rotation / translation compensation analysis using ground-truth depth.

Inputs (from GTSnapshotComponent):
  --ref    snapshot directory at the REFERENCE pose (needs depth_planar_cm.f32 + meta.json)
  --final  snapshot directory at the FINAL pose after repositioning (only meta.json is used)
  --keypoints  optional .npy, Nx2 keypoints in the reference image (OpenCV pixel convention),
               e.g. the SuperPoint keypoints that were matched at convergence.

Outputs: printed statistics + figures in --out.

Frames: everything is expressed in the reference camera's OpenCV frame
(x right, y down, z forward, metres), which avoids UE's left-handed world entirely.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------------------- loading
def load_meta(d):
    return json.loads((Path(d) / "meta.json").read_text())


def load_depth_m(d, meta):
    H, W = meta["height"], meta["width"]
    z = np.fromfile(Path(d) / "depth_planar_cm.f32", np.float32).reshape(H, W)
    return z.astype(np.float64) / 100.0


def intrinsics(meta):
    return meta["fx"], meta["fy"], meta["cx"], meta["cy"]


def cam_to_world_cv(meta):
    """Columns = CV camera axes (x right, y down, z fwd) expressed in UE world; location in metres."""
    R = np.column_stack([meta["cam_right"], -np.asarray(meta["cam_up"]), meta["cam_forward"]])
    c = np.asarray(meta["cam_location_cm"], np.float64) / 100.0
    return R, c


def relative_pose(ref, fin):
    """Final camera in the reference CV frame:  P_ref = R @ P_fin + t."""
    R_r, c_r = cam_to_world_cv(ref)
    R_f, c_f = cam_to_world_cv(fin)
    R = R_r.T @ R_f
    U, _, Vt = np.linalg.svd(R)  # re-orthonormalise (meta is written with finite precision)
    R = U @ Vt
    t = R_r.T @ (c_f - c_r)
    return R, t


# ----------------------------------------------------------------------------- geometry
def ref_points(depth, K, keypoints=None):
    """3D points in the reference CV frame + their continuous pixel coords (centre = index + 0.5)."""
    fx, fy, cx, cy = K
    H, W = depth.shape
    if keypoints is None:
        vi, ui = np.mgrid[0:H, 0:W]
        ui, vi = ui.ravel(), vi.ravel()
        uc, vc = ui + 0.5, vi + 0.5
    else:
        kp = np.asarray(keypoints, np.float64).reshape(-1, 2)  # OpenCV: centre at integer coords
        uc, vc = kp[:, 0] + 0.5, kp[:, 1] + 0.5
        ui = np.clip(np.round(kp[:, 0]).astype(int), 0, W - 1)
        vi = np.clip(np.round(kp[:, 1]).astype(int), 0, H - 1)

    Z = depth[vi, ui]
    ok = np.isfinite(Z) & (Z > 0)
    uc, vc, Z, ui, vi = uc[ok], vc[ok], Z[ok], ui[ok], vi[ok]
    P = np.column_stack([(uc - cx) / fx * Z, (vc - cy) / fy * Z, Z])
    return P, np.column_stack([uc, vc]), (vi, ui)


def project(P_ref, R, t, K):
    """Project reference-frame points into a camera at pose (R, t) relative to the reference."""
    fx, fy, cx, cy = K
    Pf = (P_ref - t) @ R  # row-wise R^T (P - t)
    return np.column_stack([fx * Pf[:, 0] / Pf[:, 2] + cx, fy * Pf[:, 1] / Pf[:, 2] + cy])


def report(name, disp, verbose=True):
    m = np.linalg.norm(disp, axis=1)
    if verbose:
        print(f"  {name:<26} mean {m.mean():8.3f}   median {np.median(m):8.3f}   "
              f"p95 {np.percentile(m, 95):8.3f}   max {m.max():8.3f}   [px]")
    return m


def stats(m):
    return {"mean": float(m.mean()), "median": float(np.median(m)),
            "p95": float(np.percentile(m, 95)), "max": float(m.max())}


# ----------------------------------------------------------------------------- analysis
def analyze(ref_dir, final_dir, out_dir, keypoints=None, max_points=20000, verbose=True,
            figures=True):
    """
    Run the full analysis of the final snapshot against the reference snapshot.
    Figures go to out_dir (unless figures=False); returns all numbers as a plain
    (YAML-friendly) dict.
    """
    log = print if verbose else (lambda *a, **k: None)

    out = Path(out_dir)
    if figures:
        out.mkdir(parents=True, exist_ok=True)

    ref, fin = load_meta(ref_dir), load_meta(final_dir)
    depth = load_depth_m(ref_dir, ref)
    K = intrinsics(ref)
    H, W = depth.shape

    R, t = relative_pose(ref, fin)
    rotvec = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(rotvec)
    axis = rotvec / theta if theta > 0 else np.array([0.0, 1.0, 0.0])

    log("\n=== Pose error (final vs reference, reference CV frame) ===")
    log(f"  rotation error      {np.degrees(theta):.4f} deg   axis (x right, y down, z fwd) = {np.round(axis, 3)}")
    log(f"  translation error   {np.linalg.norm(t):.3f} m      vector = {np.round(t, 3)}")

    kps = np.load(keypoints) if keypoints is not None else None
    P, uv, (vi, ui) = ref_points(depth, K, kps)
    log(f"  points used         {len(P)} ({'keypoints' if kps is not None else 'all valid pixels'})")

    result = {
        "rotation_error_deg": float(np.degrees(theta)),
        "rotation_axis_cv": axis.tolist(),
        "translation_error_m": float(np.linalg.norm(t)),
        "translation_vector_cv_m": t.tolist(),
        "points_used": int(len(P)),
        "point_source": "keypoints" if kps is not None else "all valid pixels",
    }

    # --- 1. depth statistics vs. effective depth implied by the errors
    Z = P[:, 2]
    Z_harm = 1.0 / np.mean(1.0 / Z)
    log("\n=== 1. Depth check ===")
    log(f"  scene depth: median {np.median(Z):.1f} m, harmonic mean {Z_harm:.1f} m, "
        f"p5-p95 {np.percentile(Z, 5):.1f}-{np.percentile(Z, 95):.1f} m")
    result["depth_m"] = {"median": float(np.median(Z)), "harmonic_mean": float(Z_harm),
                         "p5": float(np.percentile(Z, 5)), "p95": float(np.percentile(Z, 95))}
    if theta > 0:
        t_lat = np.linalg.norm(t[:2])
        log(f"  implied effective depth |t_lateral| / theta = {t_lat / theta:.1f} m")
        log(f"  predicted compensating translation theta * Z_harm = {theta * Z_harm:.3f} m "
            f"(observed lateral {t_lat:.3f} m)")
        result["depth_m"]["implied_effective"] = float(t_lat / theta)
        result["predicted_compensating_translation_m"] = float(theta * Z_harm)
        result["observed_lateral_translation_m"] = float(t_lat)

    # --- 2. decomposition
    I3, z3 = np.eye(3), np.zeros(3)
    d_full = project(P, R, t, K) - uv
    d_rot = project(P, R, z3, K) - uv
    d_tr = project(P, I3, t, K) - uv
    log("\n=== 2. Image displacement of the reference scene in the final camera ===")
    m_rot = report("rotation error only", d_rot, verbose)
    m_tr = report("translation error only", d_tr, verbose)
    m_full = report("both (actual final pose)", d_full, verbose)
    cancellation = np.median(m_full) / max(np.median(m_rot), 1e-12)
    log(f"  cancellation ratio  median(both) / median(rotation only) = {cancellation:.3f}")

    # --- 3. optimal compensating translation for the given rotation error
    rng = np.random.default_rng(0)
    sel = rng.choice(len(P), size=min(max_points, len(P)), replace=False)
    Ps, uvs = P[sel], uv[sel]
    t_star = least_squares(lambda tt: (project(Ps, R, tt, K) - uvs).ravel(), np.zeros(3)).x
    cosang = np.dot(t, t_star) / max(np.linalg.norm(t) * np.linalg.norm(t_star), 1e-12)
    angle_t_tstar = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    ratio = np.linalg.norm(t) / max(np.linalg.norm(t_star), 1e-12)
    log("\n=== 3. Best compensating translation for the observed rotation error ===")
    log(f"  t*      = {np.round(t_star, 3)}  |t*| = {np.linalg.norm(t_star):.3f} m")
    log(f"  actual  = {np.round(t, 3)}  |t|  = {np.linalg.norm(t):.3f} m")
    log(f"  angle between t and t*  {angle_t_tstar:.1f} deg, magnitude ratio |t|/|t*| = {ratio:.2f}")
    m_res = report("residual with t*", project(P, R, t_star, K) - uv, verbose)

    result["displacement_px"] = {
        "rotation_only": stats(m_rot),
        "translation_only": stats(m_tr),
        "both": stats(m_full),
        "residual_with_t_star": stats(m_res),
    }
    result["cancellation_ratio"] = float(cancellation)
    result["t_star_cv_m"] = t_star.tolist()
    result["angle_t_vs_t_star_deg"] = float(angle_t_tstar)
    result["magnitude_ratio_t_over_t_star"] = float(ratio)
    # split t into its part along the compensating direction and the rest
    u_star = t_star / max(np.linalg.norm(t_star), 1e-12)
    t_par = float(np.dot(t, u_star))
    result["translation_along_t_star_m"] = t_par
    result["translation_off_t_star_m"] = float(np.linalg.norm(t - t_par * u_star))
    result["figures"] = {}

    # --- 4. cost valley: rotation about the error axis vs translation along t*
    if figures and theta > 0 and np.linalg.norm(t_star) > 0:
        u_dir = t_star / np.linalg.norm(t_star)
        a_vals = np.linspace(-2 * theta, 2 * theta, 81)
        b_max = 2 * max(np.linalg.norm(t_star), abs(np.dot(t, u_dir)))
        b_vals = np.linspace(-b_max, b_max, 81)
        Pv, uvv = Ps[:5000], uvs[:5000]
        cost = np.empty((len(b_vals), len(a_vals)))
        for j, a in enumerate(a_vals):
            Ra = Rotation.from_rotvec(axis * a).as_matrix()
            for i, b in enumerate(b_vals):
                r = project(Pv, Ra, u_dir * b, K) - uvv
                cost[i, j] = np.sqrt(np.mean(np.sum(r * r, axis=1)))

        fig, ax = plt.subplots(figsize=(7, 5.5))
        cs = ax.contourf(np.degrees(a_vals), b_vals, np.log10(cost + 1e-6), levels=30, cmap="viridis")
        fig.colorbar(cs, label="log10 RMS reprojection error [px]")
        ax.plot(0, 0, "w+", ms=14, mew=2, label="ground truth")
        ax.plot(np.degrees(theta), np.dot(t, u_dir), "r*", ms=14, label="final pose")
        ax.plot(np.degrees(theta), np.linalg.norm(t_star), "wo", mfc="none", ms=10, label="optimal t*")
        ax.plot(np.degrees(a_vals), a_vals * Z_harm, "w--", lw=1, label=r"$t = \theta \cdot Z_{harm}$")
        ax.set_xlabel("rotation about error axis [deg]")
        ax.set_ylabel("translation along t* direction [m]")
        ax.set_title("Rotation-translation ambiguity valley")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "ambiguity_valley.png", dpi=200)
        plt.close(fig)
        result["figures"]["ambiguity_valley"] = str(out / "ambiguity_valley.png")

    # --- displacement heatmaps (dense mode only)
    if figures and kps is None:
        vmax = np.percentile(m_rot, 99)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        for axx, m, title in zip(axs, [m_rot, m_tr, m_full],
                                 ["rotation error only", "translation error only", "both (actual)"]):
            img = np.full((H, W), np.nan)
            img[vi, ui] = m
            im = axx.imshow(img, cmap="magma", vmin=0, vmax=vmax)
            axx.set_title(f"{title}\nmedian {np.nanmedian(img):.2f} px")
            axx.axis("off")
        fig.colorbar(im, ax=axs, shrink=0.8, label="pixel displacement [px]")
        fig.savefig(out / "displacement_decomposition.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        result["figures"]["displacement_decomposition"] = str(out / "displacement_decomposition.png")

    if figures:
        log(f"\nFigures written to {out.resolve()}\n")
    return result


# ----------------------------------------------------------------------------- position compensation
def load_world_xyz_m(d, meta):
    H, W = meta["height"], meta["width"]
    return np.fromfile(Path(d) / "world_xyz_cm.f64", np.float64).reshape(H, W, 3) / 100.0


def cv_axes(v):
    return {"right": float(v[0]), "down": float(v[1]), "forward": float(v[2])}


def position_compensation(ref_dir, final_dir, stride=2, clip_input_px=224):
    """
    Express the rotation error as an equivalent camera translation and add it to
    the actual translation error to get the position error the image actually shows.

      1. Rotation pixel shift from the intrinsics alone (depth independent):
             u_rot = K R^T K^-1 u
      2. Each pixel's 3D point P (reference CV frame) from depth_planar_cm.f32,
         with world_xyz_cm.f64 as a cross-check.
      3. Rotation-equivalent translation per pixel: the camera shift that moves
         P by the same amount in the image as the rotation does:
             R^T P = P - t_rot   ->   t_rot = P - R^T P     (|t_rot| ~ Z * theta)
      4. The final camera sees R^T (P - t) ~ P - (t_rot + t). Only the part
         perpendicular to the viewing ray moves the pixel, so
             visible error = perp(t + t_rot)
         If |visible| < |perp(t)|, the translation error is (partly) cancelling
         the rotation error.
    """
    ref, fin = load_meta(ref_dir), load_meta(final_dir)
    fx, fy, cx, cy = intrinsics(ref)
    H, W = ref["height"], ref["width"]
    R, t = relative_pose(ref, fin)
    theta = np.linalg.norm(Rotation.from_matrix(R).as_rotvec())

    vi, ui = np.mgrid[0:H:stride, 0:W:stride]
    vi, ui = vi.ravel(), ui.ravel()
    uv = np.column_stack([ui + 0.5, vi + 0.5])

    # --- 1. rotation pixel shift, intrinsics + rotation only
    rays = np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))])
    rr = rays @ R  # row-wise R^T ray
    uv_rot = np.column_stack([fx * rr[:, 0] / rr[:, 2] + cx, fy * rr[:, 1] / rr[:, 2] + cy])
    shift_rot_all = np.linalg.norm(uv_rot - uv, axis=1)

    # --- 2. 3D points from the .f64 world positions and the .f32 depth
    X = load_world_xyz_m(ref_dir, ref)[vi, ui]
    Z = load_depth_m(ref_dir, ref)[vi, ui]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Z) & (Z > 0)
    R_r, c_r = cam_to_world_cv(ref)
    P = rays[ok] * Z[ok, None]
    # cross-check: the same points from the world positions. Exports made before the
    # camera-scale fix in GTSnapshotComponent.cpp have their offset from the camera
    # scaled by the camera's world scale (0.1 -> ratio 10), so depth is used here.
    P_xyz = (X[ok] - c_r) @ R_r  # row-wise R_r^T (X - c)
    xyz_scale = float(np.median(np.linalg.norm(P, axis=1) / np.linalg.norm(P_xyz, axis=1)))
    xyz_dir_err = float(np.degrees(np.median(np.arccos(np.clip(
        np.sum(P * P_xyz, axis=1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(P_xyz, axis=1)),
        -1, 1)))))
    uv, rays, shift_rot = uv[ok], rays[ok], shift_rot_all[ok]
    Zp = P[:, 2]

    # --- 3./4. rotation-equivalent translation, actual translation, their sum
    n = P / np.linalg.norm(P, axis=1, keepdims=True)

    def perp(v):
        return v - np.sum(v * n, axis=1, keepdims=True) * n

    t_rot = P - P @ R
    t_rot_perp = np.linalg.norm(perp(t_rot), axis=1)
    t_perp = np.linalg.norm(perp(np.broadcast_to(t, P.shape)), axis=1)
    visible = np.linalg.norm(perp(t + t_rot), axis=1)

    # pixel shifts: translation only, and the full final pose (exact projection)
    shift_trans = np.linalg.norm(project(P, np.eye(3), t, (fx, fy, cx, cy)) - uv, axis=1)
    shift_both = np.linalg.norm(project(P, R, t, (fx, fy, cx, cy)) - uv, axis=1)

    Z_med = float(np.median(Zp))
    Z_harm = float(1.0 / np.mean(1.0 / Zp))
    clip_scale = clip_input_px / min(H, W)  # CLIP resizes the short side to 224 px
    cm = 100.0

    med_rot_px = float(np.median(shift_rot))
    med_trans_px = float(np.median(shift_trans))
    med_both_px = float(np.median(shift_both))
    med_t_rot = float(np.median(t_rot_perp)) * cm
    med_t = float(np.median(t_perp)) * cm
    med_vis = float(np.median(visible)) * cm

    return {
        "rotation_error_deg": float(np.degrees(theta)),
        "translation_error_cm": float(np.linalg.norm(t)) * cm,
        "translation_error_vector_cm": cv_axes(t * cm),
        "rotation_pixel_shift_px": {  # intrinsics + rotation only, no depth needed
            "median": med_rot_px,
            "mean": float(shift_rot.mean()),
            "image_centre": float(fx * np.tan(theta)) if theta > 0 else 0.0,
        },
        "rotation_equivalent_translation_cm": {
            "median_lateral": med_t_rot,
            "median_vector": cv_axes(np.median(t_rot, axis=0) * cm),
        },
        "translation_error_lateral_cm": med_t,
        "visible_position_error_cm": med_vis,  # |perp(t + t_rot)|, median over pixels
        "translation_cancelled_by_rotation": 1.0 - med_vis / max(med_t, 1e-12),
        "rotation_cancelled_by_translation": 1.0 - med_vis / max(med_t_rot, 1e-12),
        "pixel_shift_px": {
            "rotation_only": med_rot_px,
            "translation_only": med_trans_px,
            "both": med_both_px,
            "both_in_clip_input": med_both_px * clip_scale,
        },
        "scene_depth_m": {"median": Z_med, "harmonic_mean": Z_harm},
        "cm_per_pixel_at_median_depth": Z_med / fx * cm,
        "world_xyz_check": {  # should be 1.0 and 0.0 if the .f64 and .f32 agree
            "depth_over_world_xyz_distance": xyz_scale,
            "median_direction_error_deg": xyz_dir_err,
        },
        "pixels_used": int(len(P)),
        "image_size": [int(W), int(H)],
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--keypoints", default=None)
    ap.add_argument("--out", default="rt_ambiguity_out")
    ap.add_argument("--max-points", type=int, default=20000)
    args = ap.parse_args()

    analyze(args.ref, args.final, args.out, args.keypoints, args.max_points)


if __name__ == "__main__":
    main()
