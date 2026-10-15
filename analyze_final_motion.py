#!/usr/bin/env python3
"""
Final-pose accuracy and convergence analysis for the data/finalMotion trial sets.

Layout (one trial set = one logData/motion pair, e.g. "5_05", "5_05fog", ...):

    <root>/GroundTruth/GT<k>_<stamp>/         meta.json, depth_planar_cm.f32, MotionLog.csv, ...
    <root>/logData<set>/config_<ts>.txt       ablation config incl. target=GroundTruth/GT<k>_.../rgb.png
    <root>/logData<set>/log_<ts>.txt          "Maintained target position" (settled) or "Timed Out"
    <root>/logData<set>/dataLog_<ts>.csv      per-frame estimated errors
    <root>/logData<set>/matches_<ts>.npz      last matches (mkpts1 = keypoints in the GT image)
    <root>/motion<set>/Final<n>_<stamp>/      meta.json (= true FINAL camera pose), MotionLog.csv

Per trial
---------
1. True final pose error from the two meta.json files, in the GT camera's OpenCV frame
   (x right, y down, z forward):  rotation alpha [deg] and translation t [cm].
2. Rotation compensation. With a rotation error R left over, the image is best aligned by
   the translation t* that minimises the reprojection error of the GT scene points
   (GT depth at the matched keypoints, R fixed). For a scene at depth Z, |t*| ~ alpha * Z.
   The controller drives image error to zero, so it converges towards t*, not towards 0.
3. Effective (uncompensated) translation error  eps = t - t*, split into
      eps_lat  perpendicular to the optical axis (what the image shows),
      eps_fwd  along the optical axis (weakly observable with a monocular camera).
   eps_lat is compared to the settle dead band  settle_px * Z_harm / f  (the lateral
   offset that a 1.5 px flow threshold allows at that scene depth).
4. Start condition: initial offset (translation, rotation) of the start camera pose from
   the MotionLog against the GT pose, and the field-of-view overlap = fraction of GT image
   pixels (GT depth, sky at infinity) that are inside the start camera's image.
5. Error over time: the camera pose from every MotionLog row against the GT pose.
6. CLIP scene similarity (same model and cosine score as clip_comparison.py) of the
   Start and End captures (logData<set>/images/image{Start,End}<ts>.png) and of the
   Final snapshot rgb.png against the GT rgb.png.

Per set: success rate, time, and mean / SD / RMSE / median / max of alpha, |t|, |t*|,
|eps|, eps_lat, eps_fwd, CLIP scores, ... (settled trials and all trials).

Outputs in --out (default <root>/analysis):
    results.json   everything: definitions, GT snapshots, per-set statistics, per-trial
                   results incl. the error-over-time curve
    trials.csv, summary.csv, summary.md, figures/*.png
    trials/<set>/GT<k>_<ts>_{images,translation,rotation}.png   (clip_comparison.py's
                   GT | Start | End panel and path-vs-GT plots, one set per trial)

Usage:
    python analyze_final_motion.py
    python analyze_final_motion.py --root data/finalMotion --sets 5_05 5_05fog
    python analyze_final_motion.py --no-clip
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # render to files only, no GUI windows
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import rt_ambiguity_analysis as rt


# ============================================================
# Configuration
# ============================================================

ROOT = Path("data/finalMotion")
BASE_SET = "5_05"

# Display names, in plotting order. Sets found on disk but missing here go last.
SET_LABELS = {
    "5_05": "base (tanh, 1.5 px)",
    "5_05th1.0": "settle 1.0 px",
    "5_05th0.5": "settle 0.5 px",
    "5_05linear": "linear control",
    "5_05pd": "PD control",
    "5_05ess": "essential matrix",
    "5_05nec": "NEC",
    "5_05parallax": "parallax translation",
    "5_05rainlight": "light rain",
    "5_05rain": "rain",
    "5_05snowlight": "light snow",
    "5_05snow": "snow",
    "5_05fog": "fog",
    "5_03_sunny": "5_03 clear (PNEC)",
}

DENSE_STRIDE = 4         # pixel stride for dense depth sampling (overlap, fallback t*)
MOTION_STEP = 5          # use every n-th MotionLog row for the error-over-time curves
META_MISMATCH_CM = 5.0   # Final meta.json vs last MotionLog pose; above this the meta is stale
CLIP_MODEL = "ViT-B/32"  # as in clip_comparison.py

# dataLog rot_error_{x,y,z} -> GT CV frame rotation vector: cv = sign * est[perm]
# (fitted over all 48 signed axis permutations; the same mapping wins for every run)
EST_TO_CV = ((1, 2, 0), (-1, -1, -1))
ROT_CONVERGED_DEG = 1.0  # frames with a true rotation error below this count as "converged"
RUN_TIMEOUT_S = 180.0  # --run layout: a run this long is taken as timed out
ESTIMATOR_SETS = ["5_05", "5_05ess", "5_05nec"]  # sets that differ only in the rotation estimator
ESTIMATOR_LABELS = {"5_05": "PNEC (base)", "5_05ess": "essential matrix", "5_05nec": "NEC"}

# categorical slots 1/2 of the reference palette: settled / timed out
C_OK, C_FAIL = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

STAMP_RE = r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"


# ============================================================
# Poses
# ============================================================

def ue_axes(pitch, yaw, roll):
    """UE FRotationMatrix axes (forward, right, up) in UE world, angles in degrees."""
    p, y, r = np.radians([pitch, yaw, roll])
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return fwd, right, up


def motion_row_meta(row, cam_offset_cm):
    """A meta-like dict (camera pose) for one MotionLog row; the gimbal carries the camera."""
    fwd, right, up = ue_axes(row["PitchCam"], row["YawCam"], row["RollCam"])
    loc = np.array([row["PosX"], row["PosY"], row["PosZ"]], float) + cam_offset_cm
    return {"cam_forward": fwd, "cam_right": right, "cam_up": up, "cam_location_cm": loc}


def pose_error(ref, fin):
    """(rotation deg, translation vector cm in the ref CV frame) of fin against ref."""
    R, t = rt.relative_pose(ref, fin)
    return np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())), R, t * 100.0


# ============================================================
# Ground truth
# ============================================================

class GroundTruth:
    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.id = int(re.match(r"GT(\d+)_", path.name).group(1))
        self.meta = rt.load_meta(path)
        self.K = rt.intrinsics(self.meta)
        self.depth = rt.load_depth_m(path, self.meta)
        last = pd.read_csv(path / "MotionLog.csv").iloc[-1]
        # camera = drone position + fixed mount offset (the body never rotates, the gimbal does)
        self.cam_offset_cm = (np.asarray(self.meta["cam_location_cm"], float)
                              - last[["PosX", "PosY", "PosZ"]].to_numpy(float))

        # dense rays of the GT image, depth (inf where invalid = sky)
        H, W = self.depth.shape
        vi, ui = np.mgrid[0:H:DENSE_STRIDE, 0:W:DENSE_STRIDE]
        self.uv = np.column_stack([ui.ravel() + 0.5, vi.ravel() + 0.5])
        fx, fy, cx, cy = self.K
        self.rays = np.column_stack([(self.uv[:, 0] - cx) / fx, (self.uv[:, 1] - cy) / fy,
                                     np.ones(len(self.uv))])
        self.Z = self.depth[vi.ravel(), ui.ravel()]
        valid = np.isfinite(self.Z) & (self.Z > 0)
        self.Z_harm_dense = float(1.0 / np.mean(1.0 / self.Z[valid]))

    def overlap(self, other_meta):
        """Fraction of GT image pixels that fall inside a camera at other_meta (sky at infinity)."""
        R, t = rt.relative_pose(self.meta, other_meta)
        fx, fy, cx, cy = self.K
        W, H = self.meta["width"], self.meta["height"]
        finite = np.isfinite(self.Z) & (self.Z > 0)
        # finite points: R^T (P - t); sky: direction only, R^T ray
        P = np.where(finite[:, None], self.rays * np.where(finite, self.Z, 1.0)[:, None] - t, self.rays)
        Pc = P @ R
        u = fx * Pc[:, 0] / Pc[:, 2] + cx
        v = fy * Pc[:, 1] / Pc[:, 2] + cy
        inside = (Pc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return float(inside.mean())

    def points(self, keypoints=None):
        """3D GT-frame points + pixels for the t* fit: matched keypoints, else dense pixels."""
        if keypoints is not None and len(keypoints) >= 10:
            P, uv, _ = rt.ref_points(self.depth, self.K, keypoints)
            if len(P) >= 10:
                return P, uv, "keypoints"
        ok = np.isfinite(self.Z) & (self.Z > 0)
        return self.rays[ok] * self.Z[ok, None], self.uv[ok], "dense"


# ============================================================
# CLIP
# ============================================================

class ClipScorer:
    """Cosine similarity of CLIP image embeddings (clip_comparison.scene_similarity), cached per file."""

    def __init__(self, model_name=CLIP_MODEL):
        import torch
        import clip
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CLIP {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.cache = {}

    def embed(self, path):
        key = str(Path(path).resolve())
        if key not in self.cache:
            img = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                f = self.model.encode_image(img).float()
            self.cache[key] = self.torch.nn.functional.normalize(f, dim=-1)
        return self.cache[key]

    def similarity(self, path1, path2):
        if path1 is None or path2 is None or not Path(path1).exists() or not Path(path2).exists():
            return np.nan
        return float((self.embed(path1) @ self.embed(path2).T).item())


def load_ground_truths(root):
    gts = {}
    for p in sorted((root / "GroundTruth").iterdir()):
        if p.is_dir() and (p / "meta.json").exists():
            gt = GroundTruth(p)
            gts[gt.name] = gt
    return gts


# ============================================================
# Trial discovery
# ============================================================

def stamp_dt(s):
    return datetime.strptime(s, "%Y_%m_%d_%H_%M_%S")


def read_config(path):
    cfg = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def read_outcome(path):
    text = path.read_text(errors="replace")
    if "Maintained target position" in text:
        status = "settled"
    elif "Timed Out" in text:
        status = "timed_out"
    else:
        status = "unknown"
    m = re.search(r"timing match_id=\d+ cam_frame=\d+ n=(\d+)", text)
    return status, (int(m.group(1)) if m else np.nan)


def discover_sets(root):
    found = [p.name[len("logData"):] for p in root.glob("logData*") if p.is_dir()]
    order = list(SET_LABELS)
    return sorted(found, key=lambda s: (order.index(s) if s in order else len(order), s))


def discover_trials(root, set_name):
    """One dict per log; each Final<n> folder goes to the latest log started before it."""
    log_dir, motion_dir = root / f"logData{set_name}", root / f"motion{set_name}"
    trials = []
    for log in sorted(log_dir.glob("log_*.txt")):
        ts = re.search(STAMP_RE, log.name).group(1)
        cfg = read_config(log_dir / f"config_{ts}.txt")
        status, n_first = read_outcome(log)
        npz = log_dir / f"matches_{ts}.npz"
        trials.append({"set": set_name, "stamp": ts, "start_dt": stamp_dt(ts), "config": cfg,
                       "status": status, "n_matches_first": n_first,
                       "gt_name": Path(cfg.get("target", "")).parent.name,
                       "data_log": log_dir / f"dataLog_{ts}.csv",
                       "image_start": log_dir / "images" / f"imageStart{ts}.png",
                       "image_end": log_dir / "images" / f"imageEnd{ts}.png",
                       "matches": npz if npz.exists() else None, "final_dir": None})

    finals = []
    if motion_dir.exists():
        for d in motion_dir.iterdir():
            m = re.match(r"Final(\d+)_(\d{8}_\d{6})$", d.name)
            if m and (d / "meta.json").exists():
                finals.append((datetime.strptime(m.group(2), "%Y%m%d_%H%M%S"), int(m.group(1)), d))
    for dt, n, d in sorted(finals):
        cands = [tr for tr in trials if tr["start_dt"] <= dt and tr["final_dir"] is None]
        if not cands:
            print(f"WARNING [{set_name}] no log for {d.name}")
            continue
        tr = max(cands, key=lambda tr: tr["start_dt"])
        tr["final_dir"], tr["trial_index"] = d, n
    for tr in trials:
        if tr["final_dir"] is None:
            print(f"WARNING [{set_name}] log {tr['stamp']} has no Final snapshot (status {tr['status']})")
    return trials


def discover_trials_run(run_dir, weather, gts, settle_px=1.5):
    """
    Older run layout without logs/configs (e.g. data/motionLog5_03):
        <run>/GroundTruth/GT<k>_.../
        <run>/<weather>/finalMotion/Final<n>_<YYYYmmdd_HHMMSS>/
        <run>/<weather>/images/image{Start,End}<ts>.png
        <run>/<weather>/matches/matches_<ts>.npz
    Final<n> targets GT ((n - 1) % number of GTs) + 1. Each Final gets the latest image pair
    stamped at or before it. Without a log the outcome is unknown: runs that stopped before
    the timeout are taken as settled (status_source = "inferred").
    """
    wdir = run_dir / weather
    gt_by_id = {gt.id: name for name, gt in gts.items()}
    ids = sorted(gt_by_id)
    stamps = sorted(re.search(STAMP_RE, p.name).group(1) for p in (wdir / "images").glob("imageStart*.png"))
    set_name = f"{run_dir.name.replace('motionLog', '')}_{weather}"
    trials, used = [], set()
    for d in sorted((wdir / "finalMotion").iterdir()):
        m = re.match(r"Final(\d+)_(\d{8}_\d{6})$", d.name)
        if not m or not (d / "meta.json").exists():
            continue
        n, dt = int(m.group(1)), datetime.strptime(m.group(2), "%Y%m%d_%H%M%S")
        cands = [ts for ts in stamps if stamp_dt(ts) <= dt and ts not in used]
        if not cands:
            print(f"WARNING [{set_name}] no images for {d.name}")
            continue
        ts = cands[-1]
        used.add(ts)
        dur = pd.read_csv(d / "MotionLog.csv")["Time"].pipe(lambda t: t.iloc[-1] - t.iloc[0])
        npz = wdir / "matches" / f"matches_{ts}.npz"
        trials.append({"set": set_name, "stamp": ts, "start_dt": stamp_dt(ts),
                       "config": {"settle-px": str(settle_px), "note": "no config logged; settle-px assumed"},
                       "status": "settled" if dur < RUN_TIMEOUT_S - 1 else "timed_out",
                       "status_source": "inferred", "n_matches_first": np.nan,
                       "gt_name": gt_by_id[ids[(n - 1) % len(ids)]],
                       "data_log": wdir / f"dataLog_{ts}.csv",  # not present in this layout
                       "image_start": wdir / "images" / f"imageStart{ts}.png",
                       "image_end": wdir / "images" / f"imageEnd{ts}.png",
                       "matches": npz if npz.exists() else None,
                       "final_dir": d, "trial_index": n})
    return set_name, trials


# ============================================================
# Per-trial analysis
# ============================================================

def compensating_translation(P, uv, R, K):
    """t* [m]: translation that best re-aligns the GT points for a fixed rotation error R."""
    return least_squares(lambda tt: (rt.project(P, R, tt, K) - uv).ravel(), np.zeros(3)).x


def rms_px(P, uv, R, t, K):
    return float(np.sqrt(np.mean(np.sum((rt.project(P, R, t, K) - uv) ** 2, axis=1))))


def rotation_tracking(dl, ml, gt):
    """
    Per-frame accuracy of the algorithm's rotation estimate against the true rotation.

    True rotation error per MotionLog row: rotvec of the camera against the GT pose (GT CV
    frame, deg). The dataLog (wall clock) is aligned to the MotionLog (sim time) by the
    time offset that best matches |estimated| to |true| (log-magnitude least squares), and
    the estimate is mapped to the CV frame with EST_TO_CV (fitted once, same for all runs).
    Returns (metrics dict, per-frame array [t, |true|, |est|, |est - true|, held]).
    """
    est = dl[["rot_error_x", "rot_error_y", "rot_error_z"]].to_numpy(float)
    perm, sign = EST_TO_CV
    est = est[:, perm] * np.asarray(sign, float)
    te = dl["send_ts"].to_numpy(float) - dl["send_ts"].iloc[0]
    held = np.r_[False, np.all(np.diff(est, axis=0) == 0, axis=1)]

    tt = ml["Time"].to_numpy(float) - ml["Time"].iloc[0]
    true = np.array([np.degrees(Rotation.from_matrix(
        rt.relative_pose(gt.meta, motion_row_meta(r, gt.cam_offset_cm))[0]).as_rotvec())
        for _, r in ml.iterrows()])
    tm, em = np.linalg.norm(true, axis=1), np.linalg.norm(est, axis=1)

    offsets = np.arange(-5.0, 10.0, 0.05)
    cost = [np.mean((np.log(em + 0.05) - np.log(np.interp(te + o, tt, tm) + 0.05)) ** 2) for o in offsets]
    off = float(offsets[int(np.argmin(cost))])
    true_i = np.column_stack([np.interp(te + off, tt, true[:, j]) for j in range(3)])
    err = np.linalg.norm(est - true_i, axis=1)
    tmag = np.linalg.norm(true_i, axis=1)

    conv = tmag < ROT_CONVERGED_DEG
    step = np.linalg.norm(np.diff(est, axis=0), axis=1)
    m = {
        "rot_track_offset_s": off,
        "rot_track_align_cost": float(min(cost)),
        "rot_est_err_all_deg": float(np.median(err)),
        "rot_est_err_conv_deg": float(np.median(err[conv])) if conv.any() else np.nan,
        "rot_est_err_conv_rmse_deg": float(np.sqrt(np.mean(err[conv] ** 2))) if conv.any() else np.nan,
        "rot_est_held_frac": float(held[1:].mean()) if len(held) > 1 else np.nan,
        "rot_est_held_frac_conv": float(held[conv].mean()) if conv.any() else np.nan,
        "rot_est_step_conv_deg": float(np.mean(step[conv[1:]])) if conv[1:].any() else np.nan,
        "n_frames_conv": int(conv.sum()),
    }
    return m, np.column_stack([te + off, tmag, em, err, held])


def analyze_trial(tr, gt, settle_px):
    row = {"set": tr["set"], "stamp": tr["stamp"], "gt": gt.id, "status": tr["status"],
           "status_source": tr.get("status_source", "log"),
           "trial": tr.get("trial_index", np.nan), "n_matches_first": tr["n_matches_first"]}

    # ---- estimated errors at the end (what the algorithm believed)
    if tr["data_log"].exists():
        dl = pd.read_csv(tr["data_log"])
        if len(dl):
            last = dl.iloc[-1]
            row["est_rot_deg"] = float(np.linalg.norm(last[["rot_error_x", "rot_error_y", "rot_error_z"]]))
            row["est_px_error"] = float(last["px_error"])

    if tr["final_dir"] is None:
        return row, None, None

    fin = rt.load_meta(tr["final_dir"])
    ml = pd.read_csv(tr["final_dir"] / "MotionLog.csv")
    row["duration_s"] = float(ml["Time"].iloc[-1] - ml["Time"].iloc[0])

    track = None
    if tr["data_log"].exists() and len(dl) > 5:
        m, track = rotation_tracking(dl, ml, gt)
        row.update(m)

    # clip_comparison.py's pose errors: last MotionLog row vs last GT MotionLog row (UE world)
    gt_end, end = pd.read_csv(gt.path / "MotionLog.csv").iloc[-1], ml.iloc[-1]
    for axis in ("X", "Y", "Z"):
        row[f"ue_pos_err_{axis.lower()}_cm"] = float(end[f"Pos{axis}"] - gt_end[f"Pos{axis}"])
    for a in ("Pitch", "Yaw"):
        row[f"ue_{a.lower()}_err_deg"] = float((end[f"{a}Cam"] - gt_end[f"{a}Cam"] + 180.0) % 360.0 - 180.0)

    # consistency check of the MotionLog -> camera pose conversion against the Final meta.
    # They normally agree to < 0.01 cm; a snapshot taken after the drone was already moved
    # to the next start does not, and then the last MotionLog row is the final pose.
    ml_end = motion_row_meta(ml.iloc[-1], gt.cam_offset_cm)
    chk_rot, _, chk_t = pose_error(fin, ml_end)
    row["check_motionlog_vs_meta_cm"] = float(np.linalg.norm(chk_t))
    row["check_motionlog_vs_meta_deg"] = float(chk_rot)
    row["final_pose_source"] = "meta"
    if row["check_motionlog_vs_meta_cm"] > META_MISMATCH_CM:
        print(f"WARNING [{tr['set']}] {tr['final_dir'].name}: meta.json is "
              f"{row['check_motionlog_vs_meta_cm'] / 100:.1f} m from the last MotionLog pose, using the MotionLog")
        fin, row["final_pose_source"] = ml_end, "motionlog"

    # ---- start condition
    start = motion_row_meta(ml.iloc[0], gt.cam_offset_cm)
    rot0, _, t0 = pose_error(gt.meta, start)
    row["start_rot_deg"] = float(rot0)
    row["start_trans_cm"] = float(np.linalg.norm(t0))
    row["start_overlap"] = gt.overlap(start)

    # ---- true final error
    alpha, R, t_cm = pose_error(gt.meta, fin)
    t = t_cm / 100.0
    rotvec = Rotation.from_matrix(R).as_rotvec()
    row["alpha_deg"] = float(alpha)
    row["alpha_perp_deg"] = float(np.degrees(np.linalg.norm(rotvec[:2])))  # tilt/pan part
    row["alpha_roll_deg"] = float(np.degrees(abs(rotvec[2])))
    row["t_cm"] = float(np.linalg.norm(t_cm))
    row["t_lat_cm"] = float(np.linalg.norm(t_cm[:2]))
    row["t_fwd_cm"] = float(t_cm[2])
    row["t_x_cm"], row["t_y_cm"], row["t_z_cm"] = map(float, t_cm)

    # ---- rotation compensation with GT depth at the matched keypoints
    kps = None
    if tr["matches"] is not None:
        with np.load(tr["matches"]) as z:
            kps = np.asarray(z["mkpts1"], np.float64)  # keypoints in the GT (target) image
    P, uv, src = gt.points(kps)
    Z_harm = float(1.0 / np.mean(1.0 / P[:, 2]))
    t_star = compensating_translation(P, uv, R, gt.K)
    eps = t - t_star
    fx = gt.K[0]

    row["points_source"], row["n_points"] = src, int(len(P))
    row["Z_harm_m"] = Z_harm
    row["rot_equiv_cm"] = float(np.radians(row["alpha_perp_deg"]) * Z_harm * 100)  # alpha * Z
    row["t_star_cm"] = float(np.linalg.norm(t_star) * 100)
    row["t_star_lat_cm"] = float(np.linalg.norm(t_star[:2]) * 100)
    cos = np.dot(t[:2], t_star[:2]) / max(np.linalg.norm(t[:2]) * np.linalg.norm(t_star[:2]), 1e-12)
    row["angle_t_tstar_lat_deg"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    row["eps_cm"] = float(np.linalg.norm(eps) * 100)
    row["eps_lat_cm"] = float(np.linalg.norm(eps[:2]) * 100)
    row["eps_fwd_cm"] = float(eps[2] * 100)
    row["eps_lat_px"] = float(row["eps_lat_cm"] / 100 / Z_harm * fx)
    row["deadband_cm"] = float(settle_px * Z_harm / fx * 100)
    row["rms_px_final"] = rms_px(P, uv, R, t, gt.K)
    row["rms_px_rot_only"] = rms_px(P, uv, R, np.zeros(3), gt.K)
    row["rms_px_with_t_star"] = rms_px(P, uv, R, t_star, gt.K)

    # ---- error over time
    rows = ml.iloc[::MOTION_STEP]
    if rows.index[-1] != ml.index[-1]:
        rows = pd.concat([rows, ml.iloc[[-1]]])
    curve = []
    for _, r in rows.iterrows():
        a, _, tc = pose_error(gt.meta, motion_row_meta(r, gt.cam_offset_cm))
        curve.append((r["Time"] - ml["Time"].iloc[0], a, np.linalg.norm(tc)))
    return row, np.array(curve), track


# ============================================================
# Statistics
# ============================================================

STAT_COLS = ["alpha_deg", "t_cm", "t_lat_cm", "t_star_cm", "t_star_lat_cm", "eps_cm", "eps_lat_cm",
             "eps_fwd_cm", "eps_lat_px", "deadband_cm", "Z_harm_m", "rms_px_final", "est_rot_deg",
             "duration_s", "start_trans_cm", "start_rot_deg", "start_overlap",
             "clip_start", "clip_end", "clip_gain", "clip_final_rgb",
             "rot_est_err_all_deg", "rot_est_err_conv_deg", "rot_est_err_conv_rmse_deg",
             "rot_est_held_frac", "rot_est_held_frac_conv", "rot_est_step_conv_deg"]

DEFINITIONS = {
    "alpha_deg": "true final rotation error (GT meta vs Final meta), deg",
    "t_cm": "true final translation error |t|, cm (t_x right, t_y down, t_z forward, GT camera frame)",
    "t_star_cm": "translation that best compensates alpha in the image (GT depth at matched keypoints), cm",
    "rot_equiv_cm": "alpha_perp * Z_harm: back-of-envelope translation equivalent of the rotation error, cm",
    "eps_cm": "|t - t*|: translation error not explained by rotation compensation, cm",
    "eps_lat_cm": "lateral (perpendicular to optical axis) part of eps: visible in the image, cm",
    "eps_fwd_cm": "forward (optical axis) part of eps, signed: weakly observable, cm",
    "eps_lat_px": "eps_lat converted to pixels at the harmonic-mean keypoint depth",
    "deadband_cm": "settle-px * Z_harm / f: lateral offset the settle threshold allows, cm",
    "Z_harm_m": "harmonic-mean GT depth at the matched keypoints, m",
    "rms_px_final": "RMS reprojection displacement of the GT points in the final camera, px",
    "est_rot_deg": "rotation error the algorithm estimated at the stop (last dataLog row), deg",
    "start_overlap": "fraction of GT image pixels inside the start camera's image (sky at infinity)",
    "clip_start": f"CLIP {CLIP_MODEL} cosine similarity, Start capture vs GT rgb",
    "clip_end": f"CLIP {CLIP_MODEL} cosine similarity, End capture vs GT rgb",
    "clip_gain": "clip_end - clip_start",
    "clip_final_rgb": f"CLIP {CLIP_MODEL} cosine similarity, Final snapshot rgb.png vs GT rgb",
    "ue_pos_err_{x,y,z}_cm": "last MotionLog drone position - GT MotionLog drone position, UE world, cm "
                             "(as clip_comparison.py)",
    "ue_{pitch,yaw}_err_deg": "last MotionLog camera pitch/yaw - GT, deg (as clip_comparison.py)",
    "error_over_time": "[time since start s, rotation error deg, translation error cm] every "
                       f"{MOTION_STEP}th MotionLog row",
    "rot_est_err_all_deg": "median over frames of |estimated - true rotation error| (dataLog vs MotionLog), deg",
    "rot_est_err_conv_deg": f"same, frames with true rotation error < {ROT_CONVERGED_DEG} deg (converged phase)",
    "rot_est_err_conv_rmse_deg": "RMS of |estimated - true| over the converged frames, deg",
    "rot_est_held_frac": "fraction of frames whose rotation estimate equals the previous frame's (held value)",
    "rot_est_held_frac_conv": "same, converged frames only",
    "rot_est_step_conv_deg": "mean frame-to-frame change of the rotation estimate, converged frames, deg",
    "rot_track_offset_s": "time offset aligning the dataLog to the MotionLog (fitted per trial), s",
    "rotation_tracking": "[t s (MotionLog time), |true| deg, |estimated| deg, |est - true| deg, held 0/1], every 2nd frame",
    "rmse": "sqrt(mean(x^2)) over trials (error vs GT = 0)",
    "sd": "sample standard deviation (ddof = 1)",
}


def jsonable(x):
    """numpy / NaN -> plain JSON values (NaN -> null)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, Path):
        return x.as_posix()
    return x


def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {k: None for k in ("n", "mean", "sd", "rmse", "median", "max")}
    return {"n": int(len(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "rmse": float(np.sqrt(np.mean(x ** 2))),  # RMS about zero (= RMSE vs GT)
            "median": float(np.median(x)), "max": float(np.abs(x).max())}


def summarize_set(df):
    out = {"trials": int(len(df)),
           "n_settled": int((df["status"] == "settled").sum()),
           "success_rate": float((df["status"] == "settled").mean())}
    for subset, d in (("settled", df[df["status"] == "settled"]), ("all", df)):
        out[subset] = {c: describe(d[c]) for c in STAT_COLS if c in d}
    s = df[df["status"] == "settled"]
    if len(s) > 1:
        out["settled"]["corr_t_lat_vs_t_star_lat"] = float(np.corrcoef(s["t_lat_cm"], s["t_star_lat_cm"])[0, 1])
    return out


def fmt(d, key, unit="", nd=1):
    if d is None or d.get(key) is None:
        return "-"
    return f"{d[key]:.{nd}f}{unit}"


def markdown_table(summary):
    """All trials (a timeout is a deliberate stop: the pose at the stop is the result)."""
    lines = ["| set | settled | n | alpha [deg] mean ± SD (RMSE) | alpha max [deg] | abs(t) [cm] mean ± SD (RMSE) | "
             "abs(t*) [cm] mean | abs(eps) [cm] mean ± SD (RMSE) | eps_lat [cm] mean ± SD (RMSE) | "
             "eps_fwd [cm] mean ± SD (RMSE) | dead band [cm] | time to stop [s] | CLIP start -> end |",
             "|" + "---|" * 13]
    for name, s in summary.items():
        st = s["all"]
        def ms(k, nd=1):
            d = st.get(k)
            if d is None or d["mean"] is None:
                return "-"
            return f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f} ({d['rmse']:.{nd}f})"
        lines.append(f"| {SET_LABELS.get(name, name)} | {s['n_settled']}/{s['trials']} | "
                     f"{fmt(st.get('alpha_deg'), 'n', nd=0)} | {ms('alpha_deg', 3)} | {fmt(st.get('alpha_deg'), 'max', nd=2)} | "
                     f"{ms('t_cm')} | {fmt(st.get('t_star_cm'), 'mean')} | {ms('eps_cm')} | "
                     f"{ms('eps_lat_cm')} | {ms('eps_fwd_cm')} | {fmt(st.get('deadband_cm'), 'mean')} | "
                     f"{fmt(st.get('duration_s'), 'mean')} | "
                     f"{fmt(st.get('clip_start'), 'mean', nd=3)} -> {fmt(st.get('clip_end'), 'mean', nd=3)} |")
    return "\n".join(lines)


# ============================================================
# Figures
# ============================================================

def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def outcome_legend(ax, **kw):
    ax.scatter([], [], s=36, c=C_OK, edgecolors="white", linewidths=1, label="settled")
    ax.scatter([], [], s=40, c=C_FAIL, marker="X", edgecolors="white", linewidths=1, label="timed out")
    ax.legend(frameon=False, fontsize=9, **kw)


def scatter_outcome(ax, x, y, ok, **kw):
    ax.scatter(x[ok], y[ok], s=36, c=C_OK, edgecolors="white", linewidths=1, zorder=3, **kw)
    ax.scatter(x[~ok], y[~ok], s=40, c=C_FAIL, marker="X", edgecolors="white", linewidths=1, zorder=3, **kw)


def fig_compensation(df, out):
    """Actual lateral translation error vs the one that compensates the rotation error."""
    d = df[df["set"] == BASE_SET].dropna(subset=["t_lat_cm"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axs[0]
    ok = (d["status"] == "settled").to_numpy()
    scatter_outcome(ax, d["t_star_lat_cm"].to_numpy(), d["t_lat_cm"].to_numpy(), ok)
    lim = 1.1 * max(d["t_lat_cm"].max(), d["t_star_lat_cm"].max())
    ax.plot([0, lim], [0, lim], color=INK_MUTED, lw=1, ls="--")
    ax.text(lim * 0.97, lim * 0.93, "t = t*", color=INK_MUTED, fontsize=9, ha="right")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("rotation-compensating translation |t*| lateral [cm]")
    ax.set_ylabel("actual translation error |t| lateral [cm]")
    ax.set_title(f"{SET_LABELS.get(BASE_SET, BASE_SET)}: actual vs rotation-compensating translation",
                 fontsize=10, color=INK, loc="left")
    style(ax)

    # grouped per trial: actual lateral |t|, compensating |t*|, residual |t - t*| (vectors, not additive)
    ax = axs[1]
    d = d.sort_values(["gt", "stamp"])
    x = np.arange(len(d))
    w = 0.27
    ax.bar(x - w, d["t_lat_cm"], width=w - 0.03, color="#184f95", label="|t| actual")
    ax.bar(x, d["t_star_lat_cm"], width=w - 0.03, color=C_OK, label="|t*| compensates rotation")
    ax.bar(x + w, d["eps_lat_cm"], width=w - 0.03, color="#86b6ef", label="|t - t*| effective (eps)")
    ax.plot(x + w, d["deadband_cm"], ls="none", marker="_", ms=9, mew=2, color=INK,
            label="settle dead band (settle px at Z_harm)")
    ax.set_xticks(x, [f"GT{g}" for g in d["gt"]], fontsize=8, rotation=90)
    ax.set_ylabel("lateral translation [cm]")
    ax.set_title("Lateral error budget per trial", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "compensation_base.png", dpi=200)
    plt.close(fig)


def fig_sets(df, out, sets):
    """Strip plots per set: alpha, eps_lat, eps_fwd, |t|."""
    cols = [("alpha_deg", "rotation error alpha [deg]"),
            ("t_cm", "translation error |t| [cm]"),
            ("eps_lat_cm", "effective lateral error eps_lat [cm]"),
            ("eps_fwd_cm", "forward error eps_fwd [cm]")]
    fig, axs = plt.subplots(1, len(cols), figsize=(4 * len(cols), 0.42 * len(sets) + 1.6), sharey=True)
    rng = np.random.default_rng(0)
    y_of = {s: i for i, s in enumerate(sets)}
    for ax, (c, label) in zip(axs, cols):
        d = df.dropna(subset=[c])
        y = d["set"].map(y_of).to_numpy() + rng.uniform(-0.18, 0.18, len(d))
        ok = (d["status"] == "settled").to_numpy()
        scatter_outcome(ax, d[c].to_numpy(), y, ok)
        for s in sets:
            v = d[d["set"] == s][c]
            if len(v):
                ax.plot([v.mean()] * 2, [y_of[s] - 0.32, y_of[s] + 0.32], color=INK, lw=2)
        if c == "eps_fwd_cm":
            ax.axvline(0, color=INK_MUTED, lw=1)
        ax.set_xlabel(label)
        style(ax)
    axs[0].set_yticks(range(len(sets)), [SET_LABELS.get(s, s) for s in sets])
    axs[0].invert_yaxis()
    outcome_legend(axs[-1], loc="lower right")
    fig.suptitle("Final pose error per trial set at the stop (bar = mean of all trials)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "errors_per_set.png", dpi=200)
    plt.close(fig)


def fig_basin(df, out):
    """Start overlap / start offset vs outcome and final effective error."""
    d = df.dropna(subset=["start_overlap"])
    ok = (d["status"] == "settled").to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axs[0]
    scatter_outcome(ax, d["start_trans_cm"].to_numpy(), d["start_rot_deg"].to_numpy(), ok)
    ax.set_xlabel("start translation offset [cm]")
    ax.set_ylabel("start rotation offset [deg]")
    ax.set_title("Start offsets (all sets)", fontsize=10, color=INK, loc="left")
    outcome_legend(ax, loc="upper left")
    style(ax)

    ax = axs[1]
    scatter_outcome(ax, 100 * d["start_overlap"].to_numpy(), d["eps_lat_cm"].to_numpy(), ok)
    ax.set_xlabel("start field-of-view overlap with GT [%]")
    ax.set_ylabel("final eps_lat [cm]")
    ax.set_title("Final effective error vs start overlap", fontsize=10, color=INK, loc="left")
    style(ax)

    ax = axs[2]
    ov = 100 * d["start_overlap"].to_numpy()
    lo, hi = 5 * np.floor(ov.min() / 5), 5 * np.ceil(ov.max() / 5 + 1e-9)
    bins = np.arange(lo, hi + 0.1, 5.0)
    idx = np.digitize(ov, bins) - 1
    rate = [ok[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
    cnt = [(idx == i).sum() for i in range(len(bins) - 1)]
    labels = [f"{int(bins[i])}-{int(min(bins[i + 1], 100))}" for i in range(len(bins) - 1)]
    x = np.arange(len(labels))
    ax.bar(x, 100 * np.nan_to_num(rate), width=0.7, color=C_OK)
    for xi, r, n in zip(x, rate, cnt):
        if n:
            ax.text(xi, 100 * r + 2, f"n={n}", ha="center", fontsize=8, color=INK_MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_xlabel("start overlap [%]")
    ax.set_ylabel("settled [%]")
    ax.set_title("Success rate by start overlap (all sets)", fontsize=10, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "convergence_basin.png", dpi=200)
    plt.close(fig)


def plot_trial_figures(row, tr, gt, out_dir):
    """
    clip_comparison.py's per-trial figures:
      <name>_images.png       GT | Start | End with the CLIP similarities
      <name>_translation.png  X, Y, Z of the trial path against the GT target
      <name>_rotation.png     camera pitch, yaw of the trial path against the GT target
    Returns {kind: path} of the figures written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"GT{gt.id}_{tr['stamp']}"
    title = f"{SET_LABELS.get(tr['set'], tr['set'])} - GT{gt.id} - {tr['stamp']} ({row['status']})"
    figs = {}

    panels = [(gt.path / "rgb.png", f"Ground truth (GT{gt.id})")]
    cs, ce = row.get("clip_start", np.nan), row.get("clip_end", np.nan)
    panels.append((tr["image_start"], "Start" + (f"  (sim {cs:.4f})" if np.isfinite(cs) else "")))
    panels.append((tr["image_end"], "End" + (f"  (sim {ce:.4f}, {ce - cs:+.4f})" if np.isfinite(ce) else "")))
    if all(Path(p).exists() for p, _ in panels):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold")
        for ax, (path, label) in zip(axes, panels):
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(label, fontsize=12)
            ax.axis("off")
        fig.tight_layout()
        figs["images"] = out_dir / f"{name}_images.png"
        fig.savefig(figs["images"], dpi=150, bbox_inches="tight")
        plt.close(fig)

    if tr["final_dir"] is None:
        return figs
    ml = pd.read_csv(tr["final_dir"] / "MotionLog.csv")
    target = pd.read_csv(gt.path / "MotionLog.csv").iloc[-1]
    time = ml["Time"] - ml["Time"].iloc[0]
    for kind, cols, labels, figsize in (
            ("translation", ["PosX", "PosY", "PosZ"], ["X (cm)", "Y (cm)", "Z (cm)"], (21, 16)),
            ("rotation", ["PitchCam", "YawCam"], ["Pitch (degrees)", "Yaw (degrees)"], (21, 16))):
        fig, axes = plt.subplots(len(cols), 1, figsize=figsize)
        fig.suptitle(f"Ground Truth vs Unreal Engine Path ({kind.capitalize()}) - {title}",
                     fontsize=16, fontweight="bold")
        for ax, c, label in zip(axes, cols, labels):
            ax.plot(time, np.full(len(time), target[c]), color=C_OK, lw=2, label="Ground Truth")
            ax.plot(time, ml[c], color=C_FAIL, ls="--", lw=1.5, label="UE Path")
            ax.set_ylabel(label, fontsize=10)
            ax.legend(loc="best", frameon=False)
            style(ax)
        axes[-1].set_xlabel("Time (s)", fontsize=10)
        fig.tight_layout()
        figs[kind] = out_dir / f"{name}_{kind}.png"
        fig.savefig(figs[kind], dpi=150, bbox_inches="tight")
        plt.close(fig)
    return figs


def fig_estimators(df, tracks, out):
    """Rotation estimator comparison (sets that differ only in rot-estimator)."""
    sets = [s for s in ESTIMATOR_SETS if (df["set"] == s).any()]
    if len(sets) < 2:
        return
    fig = plt.figure(figsize=(5 * len(sets), 8.5))
    gs = fig.add_gridspec(2, len(sets))
    for j, s in enumerate(sets):
        ax = fig.add_subplot(gs[0, j])
        d = df[(df["set"] == s) & (df["gt"] == 1)].sort_values("stamp")
        if len(d) and (s, d["stamp"].iloc[0]) in tracks:
            tr = tracks[(s, d["stamp"].iloc[0])]
            ax.plot(tr[:, 0], tr[:, 1], color=INK, lw=2, label="true")
            ax.plot(tr[:, 0], tr[:, 2], color=C_OK, lw=1.2, label="estimated")
            h = tr[:, 4] > 0
            ax.scatter(tr[h, 0], tr[h, 2], s=10, color=C_FAIL, zorder=3, label="held (= previous)")
            ax.set_yscale("log")
            ax.set_ylim(0.01, None)
        ax.set_title(f"{ESTIMATOR_LABELS.get(s, s)}: GT1, first trial", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("time [s]")
        if j == 0:
            ax.set_ylabel("rotation error [deg]")
            ax.legend(frameon=False, fontsize=8)
        style(ax)

    metrics = [("rot_est_err_conv_deg", "estimate error, converged frames [deg]"),
               ("rot_est_held_frac_conv", "held frames, converged [fraction]"),
               ("alpha_deg", "final true rotation error [deg]")]
    rng = np.random.default_rng(0)
    for j, (c, label) in enumerate(metrics[:len(sets)] if len(sets) < 3 else metrics):
        ax = fig.add_subplot(gs[1, j])
        for i, s in enumerate(sets):
            d = df[df["set"] == s].dropna(subset=[c])
            ok = (d["status"] == "settled").to_numpy()
            y = i + rng.uniform(-0.18, 0.18, len(d))
            scatter_outcome(ax, d[c].to_numpy(), y, ok)
            if len(d):
                ax.plot([d[c].median()] * 2, [i - 0.32, i + 0.32], color=INK, lw=2)
        ax.set_yticks(range(len(sets)), [ESTIMATOR_LABELS.get(s, s) for s in sets])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        style(ax)
    fig.suptitle("Rotation estimators: per-frame estimate vs ground truth (bar = median)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "rotation_estimators.png", dpi=200)
    plt.close(fig)


def fig_curves(curves, df, out, set_name):
    keys = [k for k in curves if k[0] == set_name]
    if not keys:
        return
    status = df.set_index(["set", "stamp"])["status"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in keys:
        c = curves[k]
        col = C_OK if status.get(k) == "settled" else C_FAIL
        axs[0].plot(c[:, 0], c[:, 2], color=col, lw=1.2, alpha=0.8)
        axs[1].plot(c[:, 0], c[:, 1], color=col, lw=1.2, alpha=0.8)
    axs[0].set_ylabel("translation error |t| [cm]")
    axs[1].set_ylabel("rotation error [deg]")
    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel("time since start [s]")
        style(ax)
    axs[1].axhline(0.1, color=INK_MUTED, lw=1, ls="--")
    axs[1].text(axs[1].get_xlim()[1], 0.1, " 0.1 deg ", color=INK_MUTED, fontsize=8, va="bottom", ha="right")
    axs[0].plot([], [], color=C_OK, label="settled")
    axs[0].plot([], [], color=C_FAIL, label="timed out")
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"True pose error over time: {SET_LABELS.get(set_name, set_name)} "
                 f"({len(keys)} trials, random start)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"error_over_time_{set_name}.png", dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--sets", nargs="*", default=None, help="set suffixes, e.g. 5_05 5_05fog (default: all)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run", type=Path, default=None,
                    help="older run folder without logs, e.g. data/motionLog5_03 (one set per weather subfolder)")
    ap.add_argument("--settle-px", type=float, default=1.5, help="--run layout: assumed settle threshold")
    ap.add_argument("--no-clip", action="store_true", help="skip the CLIP similarities")
    ap.add_argument("--no-trial-plots", action="store_true",
                    help="skip the per-trial image panel and translation/rotation plots")
    args = ap.parse_args()
    clip_scorer = None if args.no_clip else ClipScorer()

    global BASE_SET
    if args.run is not None:
        args.root = args.run
    out = args.out or args.root / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    gts = load_ground_truths(args.root)
    if args.run is not None:
        found = [discover_trials_run(args.run, w.name, gts, args.settle_px)
                 for w in sorted(args.run.iterdir()) if (w / "finalMotion").is_dir()]
        run_trials = dict(found)
        sets = [n for n, _ in found]
    else:
        sets = args.sets or discover_sets(args.root)
    if BASE_SET not in sets:
        BASE_SET = sets[0]
    print(f"{len(gts)} GT snapshots, sets: {', '.join(sets)}")
    for gt in gts.values():
        print(f"  {gt.name}: harmonic-mean depth {gt.Z_harm_dense:.0f} m, camera offset {gt.cam_offset_cm} cm")

    rows, curves, tracks = [], {}, {}
    for s in sets:
        trials = run_trials[s] if args.run is not None else discover_trials(args.root, s)
        for tr in trials:
            gt = gts.get(tr["gt_name"])
            if gt is None:  # GT re-captured under a new folder name: match by GT number
                m = re.match(r"GT(\d+)_", tr["gt_name"])
                gt = next((g for g in gts.values() if m and g.id == int(m.group(1))), None)
            if gt is None:
                print(f"WARNING [{s}] unknown target {tr['gt_name']}")
                continue
            settle_px = float(tr["config"].get("settle-px", 1.5))
            row, curve, track = analyze_trial(tr, gt, settle_px)
            row["settle_px"] = settle_px
            if clip_scorer is not None:
                gt_rgb = gt.path / "rgb.png"
                row["clip_start"] = clip_scorer.similarity(tr["image_start"], gt_rgb)
                row["clip_end"] = clip_scorer.similarity(tr["image_end"], gt_rgb)
                row["clip_gain"] = row["clip_end"] - row["clip_start"]
                final_rgb = tr["final_dir"] / "rgb.png" if tr["final_dir"] is not None else None
                row["clip_final_rgb"] = clip_scorer.similarity(final_rgb, gt_rgb)
            row["config"] = tr["config"]
            row["files"] = {"log": tr["data_log"].with_name(f"log_{tr['stamp']}.txt"),
                            "final_dir": tr["final_dir"], "gt_dir": gt.path,
                            "image_start": tr["image_start"], "image_end": tr["image_end"],
                            "matches": tr["matches"]}
            if not args.no_trial_plots:
                row["files"].update(plot_trial_figures(row, tr, gt, out / "trials" / s))
            rows.append(row)
            if curve is not None:
                curves[(s, tr["stamp"])] = curve
            if track is not None:
                tracks[(s, tr["stamp"])] = track
        d = [r for r in rows if r["set"] == s]
        print(f"[{s}] {len(d)} trials, {sum(r['status'] == 'settled' for r in d)} settled")

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("config", "files")} for r in rows])
    ok = df["final_pose_source"] == "meta"
    chk = df.loc[ok, "check_motionlog_vs_meta_cm"].max(), df.loc[ok, "check_motionlog_vs_meta_deg"].max()
    print(f"MotionLog->camera pose check against Final meta: max {chk[0]:.4f} cm, {chk[1]:.4f} deg")
    df.to_csv(out / "trials.csv", index=False, float_format="%.4f")

    summary = {s: summarize_set(df[df["set"] == s]) for s in sets}
    for s in sets:
        cfgs = [r["config"] for r in rows if r["set"] == s]
        common = {k: v for k, v in cfgs[0].items()
                  if k != "target" and all(c.get(k) == v for c in cfgs)} if cfgs else {}
        summary[s] = {"label": SET_LABELS.get(s, s), "config": common, **summary[s]}

    trials_json = []
    for r in rows:
        tr = dict(r)
        curve = curves.get((r["set"], r["stamp"]))
        tr["error_over_time"] = curve if curve is not None else None
        track = tracks.get((r["set"], r["stamp"]))
        tr["rotation_tracking"] = track[::2] if track is not None else None
        trials_json.append(tr)
    results = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": args.root,
        "clip_model": None if args.no_clip else CLIP_MODEL,
        "definitions": DEFINITIONS,
        "ground_truths": {gt.name: {"id": gt.id, "cam_location_cm": gt.meta["cam_location_cm"],
                                    "cam_forward": gt.meta["cam_forward"],
                                    "harmonic_mean_depth_m": gt.Z_harm_dense,
                                    "cam_offset_from_drone_cm": gt.cam_offset_cm}
                          for gt in gts.values()},
        "sets": summary,
        "trials": trials_json,
    }
    (out / "results.json").write_text(json.dumps(jsonable(results), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    flat = []
    for s, v in summary.items():
        for subset in ("settled", "all"):
            for c, st in v[subset].items():
                if isinstance(st, dict):
                    flat.append({"set": s, "label": SET_LABELS.get(s, s), "subset": subset, "metric": c,
                                 "success_rate": v["success_rate"], **st})
    pd.DataFrame(flat).to_csv(out / "summary.csv", index=False, float_format="%.4f")
    md = markdown_table(summary)
    (out / "summary.md").write_text(encoding="utf-8", data=
        "All trials, error at the stop (settled or timed out). Values: mean ± SD (RMSE).\n\n" + md + "\n")

    fig_dir = out / "figures"
    fig_compensation(df, fig_dir)
    fig_sets(df, fig_dir, sets)
    fig_basin(df, fig_dir)
    fig_estimators(df, tracks, fig_dir)
    for s in sets:
        fig_curves(curves, df, fig_dir, s)

    print("\n" + md.replace("±", "+-"))
    b = df[df["set"] == BASE_SET].dropna(subset=["alpha_deg"])
    if len(b):
        print(f"\nBase set, all trials ({len(b)} trials):")
        print(f"  harmonic-mean keypoint depth  {b['Z_harm_m'].mean():.0f} m")
        print(f"  rotation error alpha          {b['alpha_deg'].mean():.3f} +- {b['alpha_deg'].std():.3f} deg")
        print(f"  alpha * Z (lateral)           {b['rot_equiv_cm'].mean():.1f} cm   (|t*| = {b['t_star_lat_cm'].mean():.1f} cm)")
        print(f"  actual lateral |t|            {b['t_lat_cm'].mean():.1f} cm")
        print(f"  effective lateral eps         {b['eps_lat_cm'].mean():.1f} +- {b['eps_lat_cm'].std():.1f} cm "
              f"(RMSE {np.sqrt((b['eps_lat_cm'] ** 2).mean()):.1f}), "
              f"= {b['eps_lat_px'].mean():.2f} px; dead band {b['deadband_cm'].mean():.1f} cm")
        print(f"  forward eps                   {b['eps_fwd_cm'].mean():.1f} +- {b['eps_fwd_cm'].std():.1f} cm")
    print(f"\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()
