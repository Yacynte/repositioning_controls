#!/usr/bin/env python3
"""
Plot real-drone repositioning runs against their GPS ground-truth target.

Two layouts are supported.

Capture layout (with images, e.g. motionLogDrone5_01):
    <root>/GroundTruthDrone/<any>/GT<k>_<ts>.png           target image
    <root>/GroundTruthDrone/<any>/*GT<k>_<ts>.csv          target motion log (last row = target pose)
    <root>/motionLogDrone<set>/Final_GT<k>/log<n>/
        motion_log_*.csv, imageStart<ts>.png, imageEnd<ts>.png, dataLog_*.csv, log_*.txt
    Per run: <out>/<set>/GT<k>_log<n>_{images,rotation,translation}.png, with the CLIP similarity
    of the Start and End image to the target image (same model as analyze_final_motion.py).
    A target position of exactly (0, 0) means no GPS (lab test); translation is then skipped.

gerolfing layout (no images, e.g. motionLogDrone5_05):
    <root>/GroundTruthDrone/gerolfing.json          list of target captures; entry i has
        images[0].gps = [lat, lon, alt], images[0].drone_att, images[0].gimbal_att = [roll, pitch, yaw]
    <root>/motionLogDrone<set>/motion_log<i>_<YYYYmmdd_HHMMSS>.csv
        Time (unix s), PosX = lat, PosY = lon, PosZ = alt [m],
        Pitch, Yaw, Roll (drone body), PitchCam, YawCam, RollCam (gimbal) [deg]
        <i> = index of the target in gerolfing.json (0-based)

Per run this writes to <out>/<set>/:
    <name>_translation.png   east / north / up offset from the target [m] and the distance
    <name>_camera_rotation.png   camera (gimbal) pitch, yaw, roll vs target
    <name>_drone_rotation.png    drone body yaw, pitch, roll vs target
and <out>/drone_results.json with the target, the time window and the error at the end of the
window (translation in m, angles in deg) for every run.

Positions are converted to a local east-north-up frame centred on the target (equirectangular
approximation, accurate to millimetres over the few hundred metres of a run).

Usage (from controls/):
    python plot_drone_runs.py
    python plot_drone_runs.py --window motion_log4_20261016_110440=0:70
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render to files only, no GUI windows
import matplotlib.pyplot as plt


ROOT = Path("data/finalMotion")
GT_FILE = "GroundTruthDrone/gerolfing.json"
EARTH_RADIUS_M = 6378137.0

# Order of the angle triplets in gerolfing.json. gimbal_att[1] is clearly the pitch (about -15 deg
# where PitchCam is about -15 deg); drone_att is assumed to use the same order.
ATT_ROLL, ATT_PITCH, ATT_YAW = 0, 1, 2

TRUNCATION_MARGIN_ROWS = 8  # rows dropped above the cut of a truncated image (partly decoded)

C_GT, C_RUN = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def wrap_deg(a):
    """Wrap angles to [-180, 180)."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def load_targets(path):
    targets = []
    for entry in json.loads(Path(path).read_text()):
        img = entry["images"][0]
        targets.append({
            "capture_id": entry.get("capture_id"),
            "time": img["time"],
            "gps": img["gps"],
            "drone_att": img["drone_att"],
            "gimbal_att": img["gimbal_att"],
            "gimbal_yaw_absolute": img.get("gimbal_yaw_absolute"),
            "images": [im["file_location"] for im in entry["images"]],
        })
    return targets


def to_enu(lat, lon, alt, lat0, lon0, alt0):
    east = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    north = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return east, north, np.asarray(alt) - alt0


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_series(path, title, time, rows, window):
    """rows: list of (label, values, target or None)."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.6 * len(rows)), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    for ax, (label, values, target) in zip(axes, rows):
        if target is not None:
            ax.axhline(target, color=C_GT, lw=2, label="target (GT)")
        ax.plot(time, values, color=C_RUN, lw=1.2, label="drone")
        ax.set_ylabel(label, fontsize=10, color=INK)
        if window is not None:
            ax.axvspan(*window, color=GRID, alpha=0.6, lw=0, label="evaluated window")
        style(ax)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    axes[-1].set_xlabel("time since log start [s]", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_run(csv_path, targets, window, out_dir):
    m = re.match(r"motion_log(\d+)_(\d{8}_\d{6})$", csv_path.stem)
    if m is None:
        print(f"WARNING: skipping {csv_path.name} (expected motion_log<index>_<YYYYmmdd_HHMMSS>.csv)")
        return None
    idx = int(m.group(1))
    tgt = targets[idx]
    lat0, lon0, alt0 = tgt["gps"]

    log = pd.read_csv(csv_path)
    t = (log["Time"] - log["Time"].iloc[0]).to_numpy()
    east, north, up = to_enu(log["PosX"], log["PosY"], log["PosZ"], lat0, lon0, alt0)
    dist = np.sqrt(east ** 2 + north ** 2 + up ** 2)

    g_yaw = tgt["drone_att"][ATT_YAW]
    g_pitch, g_roll = tgt["drone_att"][ATT_PITCH], tgt["drone_att"][ATT_ROLL]
    g_cam_pitch, g_cam_yaw = tgt["gimbal_att"][ATT_PITCH], tgt["gimbal_att"][ATT_YAW]

    title = f"{csv_path.stem}  ->  target {idx}"
    win = (0.0, float(t[-1])) if window is None else window
    shade = None if window is None else window

    plot_series(out_dir / f"{csv_path.stem}_translation.png", f"Translation vs target - {title}", t, [
        ("east [m]", east, 0.0),
        ("north [m]", north, 0.0),
        ("up [m]", up, 0.0),
        ("distance [m]", dist, 0.0),
    ], shade)
    plot_series(out_dir / f"{csv_path.stem}_camera_rotation.png", f"Camera (gimbal) rotation vs target - {title}", t, [
        ("camera pitch [deg]", log["PitchCam"], g_cam_pitch),
        ("camera yaw [deg]", log["YawCam"], g_cam_yaw),
        ("camera roll [deg]", log["RollCam"], tgt["gimbal_att"][ATT_ROLL]),
    ], shade)
    plot_series(out_dir / f"{csv_path.stem}_drone_rotation.png", f"Drone body rotation vs target - {title}", t, [
        ("drone yaw [deg]", log["Yaw"], g_yaw),
        ("drone pitch [deg]", log["Pitch"], g_pitch),
        ("drone roll [deg]", log["Roll"], g_roll),
    ], shade)

    # error at the end of the evaluated window
    i = int(np.searchsorted(t, win[1], side="right") - 1)
    i0 = int(np.searchsorted(t, win[0]))
    row = log.iloc[i]
    sel = slice(i0, i + 1)
    return {
        "log": csv_path.name,
        "target_index": idx,
        "target_gps": tgt["gps"],
        "target_drone_att": tgt["drone_att"],
        "target_gimbal_att": tgt["gimbal_att"],
        "duration_log_s": float(t[-1]),
        "window_s": [float(win[0]), float(win[1])],
        "final": {
            "time_s": float(t[i]),
            "east_m": float(east[i]), "north_m": float(north[i]), "up_m": float(up[i]),
            "horizontal_m": float(np.hypot(east[i], north[i])),
            "distance_m": float(dist[i]),
            "drone_yaw_err_deg": float(wrap_deg(row["Yaw"] - g_yaw)),
            "camera_pitch_err_deg": float(wrap_deg(row["PitchCam"] - g_cam_pitch)),
            "camera_yaw_err_deg": float(wrap_deg(row["YawCam"] - g_cam_yaw)),
        },
        "in_window": {
            "max_distance_m": float(dist[sel].max()),
            "mean_distance_m": float(dist[sel].mean()),
            "drone_yaw_err_deg_range": [float(wrap_deg(log["Yaw"].iloc[sel] - g_yaw).min()),
                                        float(wrap_deg(log["Yaw"].iloc[sel] - g_yaw).max())],
        },
        "figures": [str(out_dir / f"{csv_path.stem}_{k}.png")
                    for k in ("translation", "camera_rotation", "drone_rotation")],
    }


def find_capture_gt(root, gt_id):
    """(image, motion log) of target GT<gt_id> in any folder below <root>/GroundTruthDrone/."""
    for img in sorted((root / "GroundTruthDrone").rglob(f"GT{gt_id}_*.png")):
        logs = sorted(img.parent.glob(f"*GT{gt_id}_*.csv"))
        if logs:
            return img, logs[0]
    return None, None


def analyze_image_run(run_dir, root, out_dir, clip, targets=None):
    """
    Image-only layout (e.g. motionLogDrone5_02/<weather>/Final<...>/): log_*.txt, dataLog_*.csv,
    motion_log_*.csv and images/image{Start,End}<ts>.png. Only the image comparison is made: the
    GPS reports the drone at the target even when it is not, so the logged pose is no reference.
    The target is the image the drone actually used, named in its log ("target image: <path>"),
    searched by file name below <root>/GroundTruthDrone/.
    """
    text = "\n".join(p.read_text(errors="replace") for p in run_dir.glob("log_*.txt"))
    status = ("settled" if "Maintained target position" in text
              else "timed_out" if "Timed Out" in text else "unknown")
    m = re.search(r"target image: (\S+)", text)
    target_on_drone = m.group(1) if m else None
    target_name = Path(target_on_drone).name if target_on_drone else None
    hits = sorted((root / "GroundTruthDrone").rglob(target_name)) if target_name else []
    gt_img = hits[0] if hits else None

    start_img = next(iter(sorted(run_dir.glob("images/imageStart*.png"))), None)
    end_img = next(iter(sorted(run_dir.glob("images/imageEnd*.png"))), None)
    weather = run_dir.parent.name
    name = f"{weather}_{run_dir.name}"
    title = f"{run_dir.parent.parent.name} / {weather} / {run_dir.name}  ->  {target_name or 'unknown target'}"

    loaded = {k: load_rgb(p) if p is not None and Path(p).exists() else (None, 0, False)
              for k, p in (("gt", gt_img), ("start", start_img), ("end", end_img))}
    truncated = [str(p) for k, p in (("start", start_img), ("end", end_img)) if loaded[k][2]]
    heights = [h for _, h, _ in loaded.values() if h > 0]
    crop_h = min(heights) if heights else 0
    crops = {k: (a[:crop_h] if a is not None else None) for k, (a, _, _) in loaded.items()}
    if gt_img is None:
        print(f"WARNING: target image {target_name} of {run_dir} not found below GroundTruthDrone/ "
              f"(copy {target_on_drone} from the drone) - no CLIP score")
    ok = clip is not None and crops["gt"] is not None
    cs = clip_sim_arrays(clip, crops["start"], crops["gt"]) if ok and crops["start"] is not None else np.nan
    ce = clip_sim_arrays(clip, crops["end"], crops["gt"]) if ok and crops["end"] is not None else np.nan
    labels = ["Target (GT)" + ("" if gt_img else "  (missing)"),
              "Start" + (f"  (CLIP {cs:.4f})" if np.isfinite(cs) else ""),
              "End" + (f"  (CLIP {ce:.4f}, {ce - cs:+.4f})" if np.isfinite(ce) else "")]
    plot_image_panel(out_dir / f"{name}_images.png", title, [crops["gt"], crops["start"], crops["end"]], labels)
    figs = {"images": str(out_dir / f"{name}_images.png")}

    # ---- algorithm's own per-frame estimates (data log)
    algo = None
    data_logs = sorted(run_dir.glob("dataLog_*.csv"))
    if data_logs:
        dl = pd.read_csv(data_logs[0])
        dl.columns = dl.columns.str.strip()
        if len(dl) and "px_error" in dl:
            plot_data_log(out_dir / f"{name}_algorithm.png", f"Algorithm estimates - {title}", dl)
            figs["algorithm"] = str(out_dir / f"{name}_algorithm.png")
            rot = np.linalg.norm(dl[["rot_error_x", "rot_error_y", "rot_error_z"]].to_numpy(float), axis=1)
            n = max(1, len(dl) // 20)  # first / last 5 % of the frames
            algo = {"frames": int(len(dl)),
                    "duration_s": float(dl["send_ts"].iloc[-1] - dl["send_ts"].iloc[0]),
                    "px_error_start_median": float(dl["px_error"].iloc[:n].median()),
                    "px_error_end_median": float(dl["px_error"].iloc[-n:].median()),
                    "est_rot_error_start_median_deg": float(np.median(rot[:n])),
                    "est_rot_error_end_median_deg": float(np.median(rot[-n:]))}

    # ---- logged pose vs target pose (GPS/compass view; not the true offset)
    pose = None
    motion_logs = sorted(run_dir.glob("motion_log*.csv"))
    tgt = image_run_target_pose(run_dir, root, targets)
    if motion_logs and tgt is not None:
        log = pd.read_csv(motion_logs[0])
        t = (log["Time"] - log["Time"].iloc[0]).to_numpy()
        plot_series(out_dir / f"{name}_camera_rotation.png", f"Camera (gimbal) rotation vs target - {title}", t, [
            ("camera pitch [deg]", log["PitchCam"], tgt["PitchCam"]),
            ("camera yaw [deg]", log["YawCam"], tgt["YawCam"]),
            ("camera roll [deg]", log["RollCam"], tgt["RollCam"])], None)
        plot_series(out_dir / f"{name}_drone_rotation.png", f"Drone body rotation vs target - {title}", t, [
            ("drone yaw [deg]", log["Yaw"], tgt["Yaw"]),
            ("drone pitch [deg]", log["Pitch"], tgt["Pitch"]),
            ("drone roll [deg]", log["Roll"], tgt["Roll"])], None)
        east, north, up = to_enu(log["PosX"], log["PosY"], log["PosZ"], tgt["PosX"], tgt["PosY"], tgt["PosZ"])
        dist = np.sqrt(east ** 2 + north ** 2 + up ** 2)
        plot_series(out_dir / f"{name}_translation.png", f"Translation vs target (GPS) - {title}", t, [
            ("east [m]", east, 0.0), ("north [m]", north, 0.0), ("up [m]", up, 0.0),
            ("distance [m]", dist, 0.0)], None)
        for k in ("camera_rotation", "drone_rotation", "translation"):
            figs[k] = str(out_dir / f"{name}_{k}.png")
        pose = {"target_pose_source": tgt["source"], "duration_s": float(t[-1]),
                "start_distance_m": float(dist[0]), "end_distance_m": float(dist[-1]),
                "end_camera_yaw_err_deg": float(wrap_deg(log["YawCam"].iloc[-1] - tgt["YawCam"])),
                "end_camera_pitch_err_deg": float(wrap_deg(log["PitchCam"].iloc[-1] - tgt["PitchCam"])),
                "end_drone_yaw_err_deg": float(wrap_deg(log["Yaw"].iloc[-1] - tgt["Yaw"]))}

    return {
        "run": str(run_dir), "weather": weather, "status": status,
        "target_image_on_drone": target_on_drone,
        "target_image": str(gt_img) if gt_img else None,
        "clip_start": None if not np.isfinite(cs) else cs,
        "clip_end": None if not np.isfinite(ce) else ce,
        "truncated_images": truncated,
        "clip_crop_rows": [int(crop_h), int(loaded["start"][0].shape[0]) if loaded["start"][0] is not None else 0],
        "algorithm": algo,
        "logged_pose_vs_target": pose,
        "figures": figs,
    }


def image_run_target_pose(run_dir, root, targets):
    """
    Target pose for an image-only run, in motion-log columns. Final_idx<k> -> gerolfing entry k;
    otherwise the last row of GroundTruthDrone/Drone<set>/<weather>/*.csv.
    """
    m = re.search(r"idx(\d+)", run_dir.name)
    if m and targets and int(m.group(1)) < len(targets):
        g = targets[int(m.group(1))]
        return {"PosX": g["gps"][0], "PosY": g["gps"][1], "PosZ": g["gps"][2],
                "Roll": g["drone_att"][ATT_ROLL], "Pitch": g["drone_att"][ATT_PITCH], "Yaw": g["drone_att"][ATT_YAW],
                "RollCam": g["gimbal_att"][ATT_ROLL], "PitchCam": g["gimbal_att"][ATT_PITCH],
                "YawCam": g["gimbal_att"][ATT_YAW], "source": f"gerolfing.json entry {m.group(1)}"}
    set_name = run_dir.parent.parent.name.replace("motionLogDrone", "")
    logs = sorted((root / "GroundTruthDrone" / f"Drone{set_name}" / run_dir.parent.name).glob("*.csv"))
    if logs:
        row = pd.read_csv(logs[0]).iloc[-1]
        return {**{k: float(row[k]) for k in ("PosX", "PosY", "PosZ", "Roll", "Pitch", "Yaw",
                                              "RollCam", "PitchCam", "YawCam")}, "source": str(logs[0])}
    return None


def plot_data_log(path, title, dl):
    """Algorithm's per-frame estimates: image error, estimated rotation error, commands."""
    t = (dl["send_ts"] - dl["send_ts"].iloc[0]).to_numpy()
    xyz = (("x", "#2a78d6"), ("y", "#eb6834"), ("z", "#1baf7a"))
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")

    ax = axes[0]
    ax.plot(t, dl["px_error"], color=INK, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("image error [px]", color=INK)

    ax = axes[1]
    rot = dl[["rot_error_x", "rot_error_y", "rot_error_z"]].to_numpy(float)
    for (a, c), col in zip(xyz, rot.T):
        ax.plot(t, col, color=c, lw=1, label=a)
    ax.plot(t, np.linalg.norm(rot, axis=1), color=INK, lw=1.4, label="|rot|")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_ylabel("estimated rotation\nerror [deg]", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=4)

    for ax, prefix, label in ((axes[2], "v", "velocity command"), (axes[3], "w", "rotation-rate command")):
        for a, c in xyz:
            ax.plot(t, dl[f"{prefix}_{a}"], color=c, lw=1, label=f"{prefix}_{a}")
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        ax.set_ylabel(label, color=INK)
        ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    for ax in axes:
        style(ax)
    axes[-1].set_xlabel("time since first estimate [s]", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_rgb(path):
    """
    (RGB array, number of valid rows, truncated?) of an image. Copies off the drone can be
    truncated: such a PNG decodes only down to some row and the rest comes out black, so the
    valid height is the row where the trailing all-black block starts (minus a small margin for the
    partly written last row).
    """
    from PIL import Image, ImageFile
    try:
        arr = np.asarray(Image.open(path).convert("RGB"))
        return arr, arr.shape[0], False
    except OSError:
        pass
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        arr = np.asarray(Image.open(path).convert("RGB"))
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
    nonblack = np.flatnonzero(arr.reshape(arr.shape[0], -1).any(axis=1))
    valid = int(nonblack[-1]) + 1 if len(nonblack) else 0
    return arr, max(valid - TRUNCATION_MARGIN_ROWS, 0), True


def clip_sim_arrays(clip, a, b):
    """CLIP cosine similarity of two RGB arrays (same model/preprocessing as ClipScorer)."""
    from PIL import Image
    with clip.torch.no_grad():
        fa = clip.model.encode_image(clip.preprocess(Image.fromarray(a)).unsqueeze(0).to(clip.device)).float()
        fb = clip.model.encode_image(clip.preprocess(Image.fromarray(b)).unsqueeze(0).to(clip.device)).float()
    fa = clip.torch.nn.functional.normalize(fa, dim=-1)
    fb = clip.torch.nn.functional.normalize(fb, dim=-1)
    return float((fa @ fb.T).item())


def plot_image_panel(path, title, images, labels):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK)
    for ax, img, lab in zip(axes, images, labels):
        if img is not None:
            ax.imshow(img)
        ax.set_title(lab, fontsize=12, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_capture_run(run_dir, root, out_dir, clip):
    """
    Capture layout: <root>/motionLogDrone<set>/Final_GT<k>/log<n>/ with motion_log_*.csv,
    imageStart<ts>.png, imageEnd<ts>.png, dataLog_*.csv, log_*.txt. The target is
    <root>/GroundTruthDrone/*/GT<k>_*.png with its motion log; the last row of that log is the
    target pose. A position of exactly (0, 0) means no GPS (lab test): translation is skipped.
    """
    gt_id = int(re.search(r"Final_GT(\d+)", str(run_dir)).group(1))
    gt_img, gt_log = find_capture_gt(root, gt_id)
    if gt_img is None:
        print(f"WARNING: no target GT{gt_id} found for {run_dir}")
        return None
    tgt = pd.read_csv(gt_log).iloc[-1]
    log = pd.read_csv(sorted(run_dir.glob("motion_log*.csv"))[0])
    t = (log["Time"] - log["Time"].iloc[0]).to_numpy()
    name = f"GT{gt_id}_{run_dir.name}"
    title = f"{run_dir.parent.parent.name} / {run_dir.parent.name} / {run_dir.name}  ->  {gt_img.name}"

    text = "\n".join(p.read_text(errors="replace") for p in run_dir.glob("log_*.txt"))
    status = ("settled" if "Maintained target position" in text
              else "timed_out" if "Timed Out" in text else "unknown")
    m = re.search(r"target image: (\S+)", text)
    data_logs = list(run_dir.glob("dataLog_*.csv"))
    n_data = len(pd.read_csv(data_logs[0])) if data_logs else 0

    has_gps = not (tgt["PosX"] == 0 and tgt["PosY"] == 0)
    plot_series(out_dir / f"{name}_camera_rotation.png", f"Camera (gimbal) rotation vs target - {title}", t, [
        ("camera pitch [deg]", log["PitchCam"], tgt["PitchCam"]),
        ("camera yaw [deg]", log["YawCam"], tgt["YawCam"]),
        ("camera roll [deg]", log["RollCam"], tgt["RollCam"])], None)
    plot_series(out_dir / f"{name}_drone_rotation.png", f"Drone body rotation vs target - {title}", t, [
        ("drone yaw [deg]", log["Yaw"], tgt["Yaw"]),
        ("drone pitch [deg]", log["Pitch"], tgt["Pitch"]),
        ("drone roll [deg]", log["Roll"], tgt["Roll"])], None)

    def err(r):
        e = {"drone_yaw_err_deg": float(wrap_deg(r["Yaw"] - tgt["Yaw"])),
             "camera_pitch_err_deg": float(wrap_deg(r["PitchCam"] - tgt["PitchCam"])),
             "camera_yaw_err_deg": float(wrap_deg(r["YawCam"] - tgt["YawCam"]))}
        if has_gps:
            east, north, up = to_enu(r["PosX"], r["PosY"], r["PosZ"], tgt["PosX"], tgt["PosY"], tgt["PosZ"])
            e.update({"east_m": float(east), "north_m": float(north), "up_m": float(up),
                      "distance_m": float(np.sqrt(east ** 2 + north ** 2 + up ** 2))})
        return e

    figs = {"camera_rotation": out_dir / f"{name}_camera_rotation.png",
            "drone_rotation": out_dir / f"{name}_drone_rotation.png"}
    if has_gps:
        east, north, up = to_enu(log["PosX"], log["PosY"], log["PosZ"], tgt["PosX"], tgt["PosY"], tgt["PosZ"])
        plot_series(out_dir / f"{name}_translation.png", f"Translation vs target - {title}", t, [
            ("east [m]", east, 0.0), ("north [m]", north, 0.0), ("up [m]", up, 0.0),
            ("distance [m]", np.sqrt(east ** 2 + north ** 2 + up ** 2), 0.0)], None)
        figs["translation"] = out_dir / f"{name}_translation.png"

    start_img = next(iter(sorted(run_dir.glob("imageStart*.png"))), None)
    end_img = next(iter(sorted(run_dir.glob("imageEnd*.png"))), None)
    loaded = {k: load_rgb(p) if p is not None and Path(p).exists() else (None, 0, False)
              for k, p in (("gt", gt_img), ("start", start_img), ("end", end_img))}
    truncated = [str(p) for k, p in (("start", start_img), ("end", end_img)) if loaded[k][2]]
    full_h = loaded["gt"][0].shape[0]
    # crop all three images to the rows every one of them has, so Start and End are scored
    # against the target on exactly the same region
    crop_h = min(h for _, h, _ in loaded.values() if h > 0)
    for p in truncated:
        print(f"WARNING: {p} is truncated (incomplete copy?) - all images cropped to the top "
              f"{crop_h}/{full_h} rows ({100 * crop_h / full_h:.0f} %) for CLIP")
    crops = {k: (a[:crop_h] if a is not None else None) for k, (a, _, _) in loaded.items()}
    cs = clip_sim_arrays(clip, crops["start"], crops["gt"]) if clip and crops["start"] is not None else np.nan
    ce = clip_sim_arrays(clip, crops["end"], crops["gt"]) if clip and crops["end"] is not None else np.nan

    labels = ["Target (GT)",
              "Start" + (f"  (CLIP {cs:.4f})" if np.isfinite(cs) else ""),
              "End" + (f"  (CLIP {ce:.4f}, {ce - cs:+.4f})" if np.isfinite(ce) else "")]
    plot_image_panel(out_dir / f"{name}_images.png", title, [crops["gt"], crops["start"], crops["end"]], labels)
    figs["images"] = out_dir / f"{name}_images.png"

    return {
        "run": str(run_dir), "target": f"GT{gt_id}", "target_image": str(gt_img),
        "target_motion_log": str(gt_log), "target_image_on_drone": m.group(1) if m else None,
        "status": status, "gps": has_gps, "duration_s": float(t[-1]), "data_log_rows": n_data,
        "start": err(log.iloc[0]), "final": err(log.iloc[-1]),
        "clip_start": None if not np.isfinite(cs) else cs,
        "clip_end": None if not np.isfinite(ce) else ce,
        "truncated_images": truncated,
        "clip_crop_rows": [int(crop_h), int(full_h)],
        "figures": {k: str(v) for k, v in figs.items()},
    }


def parse_windows(items):
    windows = {}
    for item in items or []:
        name, rng = item.split("=")
        a, b = rng.split(":")
        windows[Path(name).stem] = (float(a), float(b))
    return windows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default <root>/analysisDrone")
    ap.add_argument("--window", nargs="*", default=None,
                    help="per-run evaluated time window, e.g. motion_log4_20261016_110440=0:70 (s since log start)")
    ap.add_argument("--no-clip", action="store_true", help="skip the CLIP image similarity")
    args = ap.parse_args()

    out = args.out or args.root / "analysisDrone"
    targets = load_targets(args.root / GT_FILE)
    windows = parse_windows(args.window)
    clip = None
    if not args.no_clip:
        from analyze_final_motion import ClipScorer  # same model and score as the simulation
        clip = ClipScorer()

    results = []
    for run_dir in sorted(args.root.glob("motionLogDrone*")):
        set_out = out / run_dir.name.replace("motionLogDrone", "")
        set_out.mkdir(parents=True, exist_ok=True)
        # capture layout: Final_GT<k>/log<n>/ with images
        for cap_dir in sorted(run_dir.glob("Final_GT*/log*")):
            r = analyze_capture_run(cap_dir, args.root, set_out, clip)
            if r is None:
                continue
            r["set"] = run_dir.name
            results.append(r)
            s, f = r["start"], r["final"]
            fmt_clip = lambda v: "n/a" if v is None else f"{v:.4f}"
            clip_txt = f"CLIP {fmt_clip(r['clip_start'])} -> {fmt_clip(r['clip_end'])}"
            print(f"{run_dir.name}/{cap_dir.parent.name}/{cap_dir.name}: {r['status']}, {r['duration_s']:.1f} s | "
                  f"cam yaw {s['camera_yaw_err_deg']:+.2f} -> {f['camera_yaw_err_deg']:+.2f}, "
                  f"cam pitch {s['camera_pitch_err_deg']:+.2f} -> {f['camera_pitch_err_deg']:+.2f}, "
                  f"yaw {s['drone_yaw_err_deg']:+.2f} -> {f['drone_yaw_err_deg']:+.2f} deg"
                  + (f", dist {f['distance_m']:.2f} m" if r["gps"] else ", no GPS") + f" | {clip_txt}")
        # image-only layout: <weather>/Final<...>/images/
        for img_dir in sorted(run_dir.glob("*/Final*/images")):
            r = analyze_image_run(img_dir.parent, args.root, set_out, clip, targets)
            r["set"] = run_dir.name
            results.append(r)
            fmt_clip = lambda v: "n/a" if v is None else f"{v:.4f}"
            print(f"{run_dir.name}/{img_dir.parent.parent.name}/{img_dir.parent.name}: {r['status']} | "
                  f"target {Path(r['target_image_on_drone'] or '?').name} | "
                  f"CLIP {fmt_clip(r['clip_start'])} -> {fmt_clip(r['clip_end'])}")
        # gerolfing layout: motion_log<i>_<ts>.csv directly in the folder
        for csv_path in sorted(run_dir.glob("motion_log*.csv")):
            r = analyze_run(csv_path, targets, windows.get(csv_path.stem), set_out)
            if r is None:
                continue
            r["set"] = run_dir.name
            results.append(r)
            f = r["final"]
            print(f"{csv_path.name}: target {r['target_index']}, window {r['window_s'][0]:.0f}-{r['window_s'][1]:.0f} s | "
                  f"end: dist {f['distance_m']:.2f} m (E {f['east_m']:+.2f}, N {f['north_m']:+.2f}, U {f['up_m']:+.2f}), "
                  f"yaw {f['drone_yaw_err_deg']:+.2f}, cam pitch {f['camera_pitch_err_deg']:+.2f}, "
                  f"cam yaw {f['camera_yaw_err_deg']:+.2f} deg")

    (out / "drone_results.json").write_text(json.dumps({
        "targets_file": str(args.root / GT_FILE),
        "frame": "east/north/up metres relative to the target GPS position",
        "angle_order_assumed": "gerolfing drone_att and gimbal_att = [roll, pitch, yaw]",
        "runs": results,
    }, indent=2))
    print(f"\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()
