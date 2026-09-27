"""
Evaluate a repositioning run against its ground-truth targets.

For every trial this script:
  - scores the Start and End captures against the GT capture with CLIP,
  - plots the trial's motion log against the GT motion log (translation
    and camera rotation over time),
  - saves a GT | Start | End image panel with the similarity scores,
and writes all numbers to a YAML file in RESULTS_DIR.
"""

import re
from datetime import datetime
from pathlib import Path

import torch
import clip
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render to files only, no GUI windows
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

# Folders under data/ share a common run index, e.g. groundTruths4,
# groundTruths4/imagesGT4, motionLog4_0, motionLog4_1, results4_1. RUN_INDEX
# identifies the GT target set; TEST_INDEX picks which motionLog/results
# run against that set (the "_N" suffix). Edit both per run.
RUN_INDEX = "4"
TEST_INDEX = "1"

# Ground-truth capture images, e.g. GT1_Capture_20260628_154039.png
GT_DIR = Path(f"data/groundTruths{RUN_INDEX}/imagesGT{RUN_INDEX}")

# Ground-truth motion logs, logged at the same targets as GT_DIR. Named
# MotionLog_{id}_*.csv (older runs: GT{id}_MotionLog_*.csv). The last row
# of each log is taken as the target pose.
GT_MOTION_DIR = Path(f"data/groundTruths{RUN_INDEX}")

# One subfolder per weather condition, e.g. data/motionLog4_1/sunny/. Each
# weather folder holds MotionLog_<trial>_<timestamp>.csv files directly in
# it, plus an images/ subfolder with the paired
# imageStart<timestamp>.png / imageEnd<timestamp>.png captures for every
# trial (timestamp format: YYYY_MM_DD_HH_MM_SS, shared by a Start/End pair).
MOTIONLOG_DIR = Path(f"data/motionLog{RUN_INDEX}_{TEST_INDEX}")

# Plots and image panels go to RESULTS_DIR/<weather>/.
RESULTS_DIR = Path(f"data/results{RUN_INDEX}_{TEST_INDEX}")
OUTPUT_FILE = RESULTS_DIR / "similarity_results.yaml"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# The Start/End images carry no GT id. Trials are ordered chronologically
# (by the timestamp in their filename) within each weather folder and
# numbered 1, 2, 3, ... in that order; trial N is then matched to
# GT ((N - 1) % number_of_GTs) + 1, cycling through the GT ids in order.
# Edit this if your capture script assigns targets differently.
GT_MATCHING = "cycle_by_trial_number"

# A trial's motion log is the one whose filename timestamp falls between
# that trial's image timestamp and the next trial's (the log file is
# created a few seconds after the Start/End pair is named).


# ============================================================
# Device and CLIP model
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


# ============================================================
# Helper functions: filenames
# ============================================================

def get_gt_id(filename):
    """
    Extract the integer ID from:
        GT1_Capture_20260827_132610.png

    Returns:
        1
    """

    match = re.match(r"^GT(\d+)_Capture_", filename)

    if match is None:
        return None

    return int(match.group(1))


def get_gt_motion_id(filename):
    """
    Extract the integer ID from a GT motion log:
        MotionLog_1_20260911_134402.csv     -> 1
        GT1_MotionLog_20260628_154040.csv   -> 1
    """

    match = re.match(r"^(?:GT(\d+)_MotionLog|MotionLog_(\d+))_.*\.csv$", filename)

    if match is None:
        return None

    return int(match.group(1) or match.group(2))


def parse_motionlog_image(filename):
    """
    Extract the (kind, timestamp) pair from a motionLog capture:
        imageStart2026_09_25_12_32_41.png -> ("Start", "2026_09_25_12_32_41")
        imageEnd2026_09_25_12_32_41.png   -> ("End",   "2026_09_25_12_32_41")

    Returns:
        (None, None) if the filename doesn't match.
    """

    match = re.match(
        r"^image(Start|End)(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.",
        filename,
    )

    if match is None:
        return None, None

    return match.group(1), match.group(2)


def parse_motionlog_csv(filename):
    """
    Extract the (trial, datetime) pair from a trial motion log:
        MotionLog_1_20260925_123251.csv -> (1, datetime(2026, 9, 25, 12, 32, 51))

    Returns:
        (None, None) if the filename doesn't match.
    """

    match = re.match(r"^MotionLog_(\d+)_(\d{8}_\d{6})\.csv$", filename)

    if match is None:
        return None, None

    return int(match.group(1)), datetime.strptime(match.group(2), "%Y%m%d_%H%M%S")


def image_timestamp_to_datetime(timestamp):
    return datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")


def same_file(path1, path2):
    """
    True if both files are byte-identical (catches a GT copied in as 'start').
    """

    return path1.read_bytes() == path2.read_bytes()


# ============================================================
# Helper functions: CLIP
# ============================================================

def load_image(path):
    """
    Load an image and convert it to RGB.
    """

    return preprocess(
        Image.open(path).convert("RGB")
    ).unsqueeze(0).to(device)


def scene_similarity(path1, path2):
    """
    Calculate cosine similarity between two images using CLIP.
    """

    img1 = load_image(path1)
    img2 = load_image(path2)

    with torch.no_grad():
        f1 = model.encode_image(img1)
        f2 = model.encode_image(img2)

        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)

        similarity = (f1 @ f2.T).item()

    return similarity


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 6) if values else None


# ============================================================
# Helper functions: motion logs
# ============================================================

def load_motion_log(path):
    """
    Load a MotionLog CSV and rename columns for consistency.
    Time is normalised to start at 0.
    """

    df = pd.read_csv(path)

    df = df.rename(columns={
        "Time": "time",
        "PosX": "pos_x",
        "PosY": "pos_y",
        "PosZ": "pos_z",
        "Pitch": "pitch",
        "Yaw": "yaw",
        "Roll": "roll",
        "PitchCam": "rel_pitch",
        "YawCam": "rel_yaw",
        "RollCam": "rel_roll",
    })

    df["time"] = df["time"] - df["time"].iloc[0]

    return df


def angle_difference(a, b):
    """
    Signed difference a - b in degrees, wrapped to [-180, 180).
    """

    return (a - b + 180.0) % 360.0 - 180.0


def pose_errors(gt_df, trial_df):
    """
    Final pose error of the trial relative to the GT target pose
    (last row of each log).
    """

    gt = gt_df.iloc[-1]
    final = trial_df.iloc[-1]

    dx, dy, dz = (float(final[c] - gt[c]) for c in ("pos_x", "pos_y", "pos_z"))

    return {
        "final_position_error_cm": {
            "x": round(dx, 3),
            "y": round(dy, 3),
            "z": round(dz, 3),
            "norm": round(float(np.sqrt(dx * dx + dy * dy + dz * dz)), 3),
        },
        "final_camera_error_deg": {
            "pitch": round(float(angle_difference(final["rel_pitch"], gt["rel_pitch"])), 3),
            "yaw": round(float(angle_difference(final["rel_yaw"], gt["rel_yaw"])), 3),
        },
        "duration_s": round(float(final["time"]), 3),
    }


# ============================================================
# Helper functions: plots
# ============================================================

def plot_series(axes, time, gt_df, trial_df, columns, labels):
    """
    Plot each trial column against the constant GT target value.
    """

    for ax, column, label in zip(axes, columns, labels):
        target = gt_df[column].iloc[-1]

        ax.plot(time, np.full(len(time), target), "b-", label="Ground Truth", linewidth=2)
        ax.plot(time, trial_df[column], "r--", label="UE Path", linewidth=1.5, alpha=0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=10)


def plot_pose_comparison(gt_df, trial_df, trans_file, rot_file, title):
    """
    Save translation (x, y, z) and camera rotation (pitch, yaw) plots of
    the trial motion log against the GT target pose.
    """

    time = trial_df["time"]

    fig, axes = plt.subplots(3, 1, figsize=(21, 16))
    fig.suptitle(f"Ground Truth vs Unreal Engine Path (Translation) - {title}",
                 fontsize=16, fontweight="bold")
    plot_series(axes, time, gt_df, trial_df,
                ["pos_x", "pos_y", "pos_z"], ["X (cm)", "Y (cm)", "Z (cm)"])
    fig.tight_layout()
    fig.savefig(trans_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(21, 16))
    fig.suptitle(f"Ground Truth vs Unreal Engine Path (Rotation) - {title}",
                 fontsize=16, fontweight="bold")
    plot_series(axes, time, gt_df, trial_df,
                ["rel_pitch", "rel_yaw"], ["Pitch (degrees)", "Yaw (degrees)"])
    fig.tight_layout()
    fig.savefig(rot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_image_comparison(gt_path, start_path, end_path, start_sim, end_sim,
                          output_file, title):
    """
    Save a GT | Start | End panel with the CLIP similarities in the titles.
    """

    panels = [
        (gt_path, f"Ground truth ({gt_path.stem.split('_')[0]})"),
        (start_path, f"Start  (sim {start_sim:.4f})"),
        (end_path, f"End  (sim {end_sim:.4f}, {end_sim - start_sim:+.4f})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for ax, (path, label) in zip(axes, panels):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Find ground-truth images
# ============================================================

print("\nScanning ground-truth images...")

if not GT_DIR.exists():
    raise FileNotFoundError(
        f"GT_DIR does not exist: {GT_DIR}\n"
        f"Update GT_DIR at the top of this script to where the "
        f"GT{{id}}_Capture_*.png images actually live."
    )

gt_images = {}

for path in GT_DIR.iterdir():

    if not path.is_file():
        continue

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    image_id = get_gt_id(path.name)

    if image_id is None:
        print(f"WARNING: Could not parse GT ID: {path}")
        continue

    if image_id in gt_images:
        raise RuntimeError(
            f"Duplicate GT ID {image_id}:\n"
            f"  {gt_images[image_id]}\n"
            f"  {path}"
        )

    gt_images[image_id] = path

num_gt = len(gt_images)

print(f"Found {num_gt} GT images.")

if num_gt == 0:
    raise RuntimeError(f"No GT images found in {GT_DIR}; nothing to compare against.")


# ============================================================
# Find ground-truth motion logs
# ============================================================

gt_motion_logs = {}

if GT_MOTION_DIR.exists():

    for path in GT_MOTION_DIR.glob("*.csv"):

        motion_id = get_gt_motion_id(path.name)

        if motion_id is None:
            print(f"WARNING: Could not parse GT motion log ID: {path}")
            continue

        if motion_id in gt_motion_logs:
            raise RuntimeError(
                f"Duplicate GT motion log ID {motion_id}:\n"
                f"  {gt_motion_logs[motion_id]}\n"
                f"  {path}"
            )

        gt_motion_logs[motion_id] = path

print(f"Found {len(gt_motion_logs)} GT motion logs.")

if not gt_motion_logs:
    print(f"WARNING: No GT motion logs in {GT_MOTION_DIR}; pose plots will be skipped.")


# ============================================================
# Find weather conditions
# ============================================================

if not MOTIONLOG_DIR.exists():
    raise FileNotFoundError(f"MOTIONLOG_DIR does not exist: {MOTIONLOG_DIR}")

weather_conditions = sorted(
    p.name for p in MOTIONLOG_DIR.iterdir()
    if p.is_dir() and (p / "images").is_dir()
)

print(f"\nFound {len(weather_conditions)} weather condition(s): {weather_conditions}")


# ============================================================
# Find and pair Start/End trial images (and motion logs) per weather
# ============================================================

# trials[weather] = [(timestamp, start_path, end_path, log_path), ...]
# sorted by time; log_path is None when no motion log matches the trial
trials_by_weather = {}

for weather in weather_conditions:

    images_dir = MOTIONLOG_DIR / weather / "images"

    starts = {}
    ends = {}

    for path in images_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        kind, timestamp = parse_motionlog_image(path.name)

        if kind is None:
            print(f"WARNING: Could not parse motionLog image: {path}")
            continue

        bucket = starts if kind == "Start" else ends

        if timestamp in bucket:
            raise RuntimeError(
                f"Duplicate image{kind}{timestamp} in {images_dir}:\n"
                f"  {bucket[timestamp]}\n"
                f"  {path}"
            )

        bucket[timestamp] = path

    missing_end = sorted(set(starts) - set(ends))
    missing_start = sorted(set(ends) - set(starts))

    for timestamp in missing_end:
        print(f"WARNING: [{weather}] Start image with no matching End: {timestamp}")

    for timestamp in missing_start:
        print(f"WARNING: [{weather}] End image with no matching Start: {timestamp}")

    timestamps = sorted(set(starts) & set(ends))

    # Motion logs, sorted by their filename timestamp
    motion_logs = []

    for path in (MOTIONLOG_DIR / weather).glob("*.csv"):

        log_trial, log_time = parse_motionlog_csv(path.name)

        if log_trial is None:
            print(f"WARNING: Could not parse motion log: {path}")
            continue

        motion_logs.append((log_time, path))

    motion_logs.sort()

    # Match each trial to the log created between its images and the next trial's
    trial_logs = []

    for i, timestamp in enumerate(timestamps):

        begin = image_timestamp_to_datetime(timestamp)
        end = (image_timestamp_to_datetime(timestamps[i + 1])
               if i + 1 < len(timestamps) else datetime.max)

        matches = [path for log_time, path in motion_logs if begin <= log_time < end]

        if len(matches) > 1:
            print(f"WARNING: [{weather}] {len(matches)} motion logs match trial "
                  f"{timestamp}, using {matches[0].name}")

        trial_logs.append(matches[0] if matches else None)

    matched = {p for p in trial_logs if p is not None}

    for _, path in motion_logs:
        if path not in matched:
            print(f"WARNING: [{weather}] Motion log not matched to any trial: {path.name}")

    trials_by_weather[weather] = [
        (timestamp, starts[timestamp], ends[timestamp], log_path)
        for timestamp, log_path in zip(timestamps, trial_logs)
    ]

    print(f"  [{weather}] {len(timestamps)} complete Start/End trial(s), "
          f"{len(matched)} with a motion log")


# ============================================================
# Calculate similarities and plot each trial
# ============================================================

results = []

total_trials = sum(len(t) for t in trials_by_weather.values())
print(f"\nEvaluating {total_trials} trial(s)...\n")

trial_counter = 0

for weather in weather_conditions:

    weather_results_dir = RESULTS_DIR / weather
    weather_results_dir.mkdir(parents=True, exist_ok=True)

    for trial_index, (timestamp, start_path, end_path, log_path) in enumerate(
        trials_by_weather[weather], start=1
    ):

        trial_counter += 1

        gt_id = ((trial_index - 1) % num_gt) + 1
        gt_path = gt_images.get(gt_id)

        print(
            f"[{trial_counter}/{total_trials}] "
            f"{weather} | trial {trial_index} -> GT{gt_id} | {timestamp}"
        )

        if gt_path is None:
            print(f"    WARNING: GT{gt_id} not found, skipping trial")
            continue

        if same_file(gt_path, start_path):
            print("    WARNING: start image is identical to the GT image")

        if log_path is not None:
            log_trial, _ = parse_motionlog_csv(log_path.name)
            if log_trial != gt_id:
                print(f"    WARNING: motion log {log_path.name} is numbered "
                      f"{log_trial}, but the trial is matched to GT{gt_id}")

        # --- CLIP similarity ---

        start_sim = scene_similarity(gt_path, start_path)
        end_sim = scene_similarity(gt_path, end_path)
        improvement = end_sim - start_sim

        print(f"    Start similarity: {start_sim:.6f}")
        print(f"    End similarity:   {end_sim:.6f}")
        print(f"    Improvement:      {improvement:+.6f}")

        title = f"{weather} | trial {trial_index} | GT{gt_id} | {timestamp}"
        stem = f"trial{trial_index}_GT{gt_id}_{timestamp}"

        images_plot = weather_results_dir / f"images_{stem}.png"
        plot_image_comparison(gt_path, start_path, end_path, start_sim, end_sim,
                              images_plot, title)

        # --- Motion log vs GT motion log ---

        motion = None
        gt_log_path = gt_motion_logs.get(gt_id)

        if log_path is None:
            print("    WARNING: no motion log for this trial, skipping pose plots")
        elif gt_log_path is None:
            print(f"    WARNING: no GT motion log for GT{gt_id}, skipping pose plots")
        else:
            gt_df = load_motion_log(gt_log_path)
            trial_df = load_motion_log(log_path)

            trans_plot = weather_results_dir / f"comparison_plot_Trans_{stem}.png"
            rot_plot = weather_results_dir / f"comparison_plot_Rot_{stem}.png"
            plot_pose_comparison(gt_df, trial_df, trans_plot, rot_plot, title)

            motion = {
                "gt_motion_log": str(gt_log_path),
                "motion_log": str(log_path),
                **pose_errors(gt_df, trial_df),
                "plots": {
                    "translation": str(trans_plot),
                    "rotation": str(rot_plot),
                },
            }

            print(f"    Final position error: "
                  f"{motion['final_position_error_cm']['norm']:.1f} cm")

        results.append({
            "weather": weather,
            "trial": trial_index,
            "timestamp": timestamp,
            "gt_id": gt_id,

            "ground_truth": {
                "filename": gt_path.name,
                "path": str(gt_path),
            },

            "start": {
                "filename": start_path.name,
                "path": str(start_path),
                "similarity_to_gt": round(start_sim, 6),
            },

            "end": {
                "filename": end_path.name,
                "path": str(end_path),
                "similarity_to_gt": round(end_sim, 6),
            },

            "improvement": round(improvement, 6),
            "images_plot": str(images_plot),
            "motion": motion,
        })


# ============================================================
# Write YAML
# ============================================================

def summarize(rows):
    return {
        "mean_start_similarity": mean([r["start"]["similarity_to_gt"] for r in rows]),
        "mean_end_similarity": mean([r["end"]["similarity_to_gt"] for r in rows]),
        "mean_improvement": mean([r["improvement"] for r in rows]),
        "mean_final_position_error_cm": mean(
            [r["motion"]["final_position_error_cm"]["norm"] for r in rows if r["motion"]]),
    }


output = {
    "model": "ViT-B/32",
    "metric": "cosine_similarity",
    "gt_images_dir": str(GT_DIR),
    "gt_motion_dir": str(GT_MOTION_DIR),
    "motion_log_dir": str(MOTIONLOG_DIR),
    "gt_matching": GT_MATCHING,
    "weather_conditions": weather_conditions,
    "number_of_trials": len(results),
    "summary": {
        "overall": summarize(results),
        "by_weather": {
            weather: summarize([r for r in results if r["weather"] == weather])
            for weather in weather_conditions
        },
    },
    "trials": results,
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        output,
        file,
        sort_keys=False,
        allow_unicode=True
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Finished!")
print("=" * 60)

print(f"GT images:          {num_gt}")
print(f"Weather conditions: {len(weather_conditions)}")
print(f"Trials compared:    {len(results)}")
print(f"With pose plots:    {sum(1 for r in results if r['motion'])}")

print(f"\nResults written to:")
print(f"  {OUTPUT_FILE}")
print(f"  {RESULTS_DIR}/<weather>/  (image panels and pose plots)")
