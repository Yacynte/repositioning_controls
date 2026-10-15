"""
Evaluate a repositioning run against its ground-truth snapshots, combining
the CLIP image comparison (clip_comparison.py) with the depth-based
rotation/translation analysis (rt_ambiguity_analysis.py).

Expected layout (e.g. data/motionLog5_01):

    <RUN_DIR>/GroundTruth/GT<id>_<YYYYmmdd_HHMMSS>/
        rgb.png, depth_planar_cm.f32, world_xyz_cm.f64, meta.json, MotionLog.csv
    <RUN_DIR>/<weather>/finalMotion/Final<n>_<YYYYmmdd_HHMMSS>/
        rgb.png, depth_planar_cm.f32, world_xyz_cm.f64, meta.json, [MotionLog.csv]
    <RUN_DIR>/<weather>/images/
        imageStart<YYYY_mm_dd_HH_MM_SS>.png, imageEnd<YYYY_mm_dd_HH_MM_SS>.png

For every trial (= Final snapshot) this script:
  - scores the Start and End images against the GT rgb with CLIP,
  - saves a GT | Start | End image panel,
  - computes the pose error from the two meta.json files (Final = End pose),
  - converts the rotation error to a pixel shift (intrinsics) and then to an
    equivalent camera translation (depth), adds it to the translation error
    and decides whether the translation compensates the rotation
    (rt_ambiguity_analysis.position_compensation, GT and Final depth maps),
  - plots the trial motion log against the GT one when both exist,
and writes all numbers to RESULTS_DIR/similarity_results.yaml.

Usage:
    python evaluate_repositioning.py                       # uses RUN_DIR below
    python evaluate_repositioning.py --run data/motionLog5_01 --results data/results5_01
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")  # render to files only, no GUI windows
import matplotlib.pyplot as plt
from PIL import Image

import rt_ambiguity_analysis as rt


# ============================================================
# Configuration
# ============================================================

RUN_DIR = Path("data/motionLog5_03")
RESULTS_DIR = Path("data/results5_03")

GT_SUBDIR = "GroundTruth"
FINAL_SUBDIR = "finalMotion"
IMAGES_SUBDIR = "images"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Final<n> is matched to the n-th GT, cycling: GT ids sorted, index (n - 1) % num_gt.
GT_MATCHING = "cycle_by_trial_number"

# A Start/End image pair belongs to the Final snapshot taken right after it:
# the pair with the latest timestamp <= the Final folder's timestamp.
CLIP_MODEL = "ViT-B/32"

# Compensation verdict (rt_ambiguity_analysis.position_compensation):
#   rotation pixel shift below ROT_NEGLIGIBLE_PX       -> rotation_negligible
#   visible error < COMPENSATING_RATIO * both of
#   |translation| and |rotation-equivalent translation| -> compensating
#   visible error < |translation| only (overshoot)      -> overcompensating
#   otherwise                                           -> not_compensating
# (all lateral = perpendicular to the viewing ray, medians over pixels)
ROT_NEGLIGIBLE_PX = 1.0
COMPENSATING_RATIO = 0.9

# ============================================================
# CLIP
# ============================================================

class ClipScorer:

    def __init__(self, model_name=CLIP_MODEL):
        import torch
        import clip

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self._cache = {}

    def embed(self, path):
        path = Path(path)
        if path not in self._cache:
            img = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                f = self.model.encode_image(img)
                self._cache[path] = self.torch.nn.functional.normalize(f, dim=-1)
        return self._cache[path]

    def similarity(self, path1, path2):
        """Cosine similarity between two images' CLIP embeddings."""
        return (self.embed(path1) @ self.embed(path2).T).item()


# ============================================================
# Helper functions: discovery
# ============================================================

def parse_snapshot_dir(name, tag):
    """
    GT3_20261008_130313    -> (3, datetime(2026, 10, 8, 13, 3, 13))
    Final12_20261008_151857 -> (12, ...)
    """
    match = re.match(rf"^{tag}(\d+)_(\d{{8}}_\d{{6}})$", name)
    if match is None:
        return None, None
    return int(match.group(1)), datetime.strptime(match.group(2), "%Y%m%d_%H%M%S")


def parse_motionlog_image(filename):
    """
    imageStart2026_10_08_15_07_09.png -> ("Start", datetime(2026, 10, 8, 15, 7, 9))
    """
    match = re.match(r"^image(Start|End)(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.", filename)
    if match is None:
        return None, None
    return match.group(1), datetime.strptime(match.group(2), "%Y_%m_%d_%H_%M_%S")


def find_snapshots(directory, tag):
    """{index: (datetime, path)} for every <tag><index>_<timestamp>/ folder."""
    snapshots = {}
    if not directory.is_dir():
        return snapshots

    for path in sorted(directory.iterdir()):
        if not path.is_dir():
            continue
        index, when = parse_snapshot_dir(path.name, tag)
        if index is None:
            print(f"WARNING: Could not parse snapshot folder: {path}")
            continue
        if not (path / "meta.json").is_file():
            print(f"WARNING: Snapshot without meta.json, skipped: {path}")
            continue
        if index in snapshots:
            raise RuntimeError(f"Duplicate {tag}{index}:\n  {snapshots[index][1]}\n  {path}")
        snapshots[index] = (when, path)

    return snapshots


def find_image_pairs(images_dir):
    """Sorted [(datetime, start_path, end_path), ...] of complete Start/End pairs."""
    starts, ends = {}, {}
    if not images_dir.is_dir():
        return []

    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        kind, when = parse_motionlog_image(path.name)
        if kind is None:
            print(f"WARNING: Could not parse motionLog image: {path}")
            continue
        (starts if kind == "Start" else ends)[when] = path

    for when in sorted(set(starts) ^ set(ends)):
        print(f"WARNING: Unpaired Start/End image at {when:%Y_%m_%d_%H_%M_%S} in {images_dir}")

    return [(when, starts[when], ends[when]) for when in sorted(set(starts) & set(ends))]


def match_image_pairs(finals, pairs):
    """
    {final_index: (start_path, end_path)} - each Final gets the latest unused
    pair taken at or before it.
    """
    matched, used = {}, set()

    for index in sorted(finals, key=lambda i: finals[i][0]):
        when = finals[index][0]
        candidates = [k for k, (t, _, _) in enumerate(pairs) if t <= when and k not in used]
        if not candidates:
            print(f"WARNING: No Start/End images before Final{index}")
            continue
        k = candidates[-1]
        used.add(k)
        matched[index] = pairs[k][1:]

    for k, (t, _, _) in enumerate(pairs):
        if k not in used:
            print(f"WARNING: Start/End pair {t:%Y_%m_%d_%H_%M_%S} not matched to any Final")

    return matched


# ============================================================
# Helper functions: pose / motion logs
# ============================================================

def angle_difference(a, b):
    """Signed difference a - b in degrees, wrapped to [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def meta_pose_error(gt_meta, final_meta):
    """Final camera position error in UE world (cm), from the snapshot metadata."""
    d = np.subtract(final_meta["cam_location_cm"], gt_meta["cam_location_cm"])
    return {
        "x": round(float(d[0]), 3),
        "y": round(float(d[1]), 3),
        "z": round(float(d[2]), 3),
        "norm": round(float(np.linalg.norm(d)), 3),
    }


def load_motion_log(path):
    """Load a MotionLog CSV with consistent column names; time starts at 0."""
    import pandas as pd

    df = pd.read_csv(path).rename(columns={
        "Time": "time", "PosX": "pos_x", "PosY": "pos_y", "PosZ": "pos_z",
        "Pitch": "pitch", "Yaw": "yaw", "Roll": "roll",
        "PitchCam": "rel_pitch", "YawCam": "rel_yaw", "RollCam": "rel_roll",
    })
    df["time"] = df["time"] - df["time"].iloc[0]
    return df


def plot_series(axes, time, gt_df, trial_df, columns, labels):
    """Plot each trial column against the constant GT target value."""
    for ax, column, label in zip(axes, columns, labels):
        target = gt_df[column].iloc[-1]
        ax.plot(time, np.full(len(time), target), "b-", label="Ground Truth", linewidth=2)
        ax.plot(time, trial_df[column], "r--", label="UE Path", linewidth=1.5, alpha=0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)", fontsize=10)


def plot_pose_comparison(gt_df, trial_df, trans_file, rot_file, title):
    """Translation (x, y, z) and camera rotation (pitch, yaw) of the trial vs the GT target."""
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


# ============================================================
# Helper functions: image panel
# ============================================================

def plot_panel(gt_dir, start_path, end_path, sims, output_file, title):
    """GT | Start | End, with the CLIP similarities to the GT in the titles."""
    panels = [
        (gt_dir / "rgb.png", f"Ground truth ({gt_dir.name.split('_')[0]})"),
        (start_path, f"Start  (sim {sims['start']:.4f})"),
        (end_path, f"End  (sim {sims['end']:.4f}, {sims['end'] - sims['start']:+.4f})"),
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
# Helper functions: rotation/translation compensation
# ============================================================

def r3(v):
    return round(float(v), 3)


def round_dict(d):
    return {k: round_dict(v) if isinstance(v, dict) else r3(v) for k, v in d.items()}


def classify(pc):
    vis = pc["visible_position_error_cm"]
    t_lat = pc["translation_error_lateral_cm"]
    rot_lat = pc["rotation_equivalent_translation_cm"]["median_lateral"]

    if pc["rotation_pixel_shift_px"]["median"] < ROT_NEGLIGIBLE_PX:
        return "rotation_negligible"
    if vis < COMPENSATING_RATIO * min(t_lat, rot_lat):
        return "compensating"
    if vis < t_lat:
        return "overcompensating"
    return "not_compensating"


VERDICT_TEXT = {
    "compensating":
        "the translation error points opposite to the rotation-equivalent translation, "
        "so the two cancel and the image shows a much smaller error than either alone",
    "overcompensating":
        "the translation error opposes the rotation error but is larger than needed: "
        "part of it is cancelled, the rest is visible",
    "not_compensating":
        "the translation error does not cancel the rotation error; the visible error "
        "is at least as large as the translation error",
    "rotation_negligible":
        f"the rotation error shifts the image by less than {ROT_NEGLIGIBLE_PX} px, "
        "so there is nothing for the translation to compensate",
}


def clip_explanation(pc):
    """One paragraph tying the pose error to the pixel shift CLIP sees."""
    rot = pc["rotation_equivalent_translation_cm"]["median_lateral"]
    px = pc["pixel_shift_px"]
    return (
        f"Rotation error {pc['rotation_error_deg']:.3f} deg shifts the image by "
        f"{pc['rotation_pixel_shift_px']['median']:.2f} px (from K and R only), which at the "
        f"scene depth (median {pc['scene_depth_m']['median']:.0f} m, "
        f"{pc['cm_per_pixel_at_median_depth']:.1f} cm per pixel) equals a camera shift of "
        f"{rot:.1f} cm. The translation error is {pc['translation_error_cm']:.1f} cm "
        f"({pc['translation_error_lateral_cm']:.1f} cm lateral). Added together they leave a "
        f"visible position error of {pc['visible_position_error_cm']:.1f} cm, i.e. "
        f"{px['both']:.2f} px in the {pc['image_size'][0]}x{pc['image_size'][1]} image and "
        f"{px['both_in_clip_input']:.2f} px at CLIP's 224 px input"
        + (" - below one CLIP input pixel, so invisible to the CLIP embedding."
           if px['both_in_clip_input'] < 1.0 else ".")
    )


def compensation_report(pc_gt, pc_final, rt_gt):
    """
    pc_gt    - position_compensation with the GT depth map (GT scene seen from the End pose)
    pc_final - same with the Final depth map (End scene seen from the GT pose)
    rt_gt    - rt.analyze with the GT depth map (optimal compensating translation t*)
    """
    verdict, verdict_final = classify(pc_gt), classify(pc_final)
    vec = pc_gt["translation_error_vector_cm"]
    rot_vec = pc_gt["rotation_equivalent_translation_cm"]["median_vector"]

    return {
        "verdict": verdict,
        "explanation": VERDICT_TEXT[verdict],
        "is_compensating": verdict in ("compensating", "overcompensating"),
        "clip_explanation": clip_explanation(pc_gt),

        "rotation_error_deg": r3(pc_gt["rotation_error_deg"]),
        "rotation_pixel_shift_px": round_dict(pc_gt["rotation_pixel_shift_px"]),
        "scene_depth_m": round_dict(pc_gt["scene_depth_m"]),
        "cm_per_pixel_at_median_depth": r3(pc_gt["cm_per_pixel_at_median_depth"]),

        # vectors in the GT camera frame (right / down / forward), cm
        "translation_error_cm": round_dict(vec),
        "rotation_equivalent_translation_cm": round_dict(rot_vec),
        "translation_plus_rotation_equivalent_cm": {
            k: r3(vec[k] + rot_vec[k]) for k in vec},

        # medians over pixels of the part perpendicular to each viewing ray
        "lateral_cm": {
            "translation_error": r3(pc_gt["translation_error_lateral_cm"]),
            "rotation_equivalent": r3(pc_gt["rotation_equivalent_translation_cm"]["median_lateral"]),
            "visible_error": r3(pc_gt["visible_position_error_cm"]),
        },
        "translation_error_cancelled_fraction": r3(pc_gt["translation_cancelled_by_rotation"]),
        "pixel_shift_px": round_dict(pc_gt["pixel_shift_px"]),

        "optimal_compensating_translation": {
            "t_star_norm_cm": r3(np.linalg.norm(rt_gt["t_star_cv_m"]) * 100),
            "angle_translation_vs_t_star_deg": r3(rt_gt["angle_t_vs_t_star_deg"]),
            "best_possible_pixel_shift_px": r3(rt_gt["displacement_px"]["residual_with_t_star"]["median"]),
        },

        "final_depth_check": {
            "verdict": verdict_final,
            "agrees": verdict == verdict_final,
            "visible_error_cm": r3(pc_final["visible_position_error_cm"]),
            "pixel_shift_both_px": r3(pc_final["pixel_shift_px"]["both"]),
        },
        "world_xyz_check": round_dict(pc_gt["world_xyz_check"]),
    }


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 6) if values else None


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN_DIR, help="run folder, e.g. data/motionLog5_01")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR, help="output folder")
    ap.add_argument("--max-points", type=int, default=20000, help="points for the t* fit")
    ap.add_argument("--no-clip", action="store_true", help="skip CLIP similarities")
    ap.add_argument("--no-depth", action="store_true", help="skip the compensation analysis")
    ap.add_argument("--verbose", action="store_true", help="print the full rt analysis per trial")
    args = ap.parse_args()

    run_dir, results_dir = args.run, args.results

    # --- ground truth
    gt_snapshots = find_snapshots(run_dir / GT_SUBDIR, "GT")
    if not gt_snapshots:
        raise RuntimeError(f"No GT<id>_<timestamp>/ snapshots in {run_dir / GT_SUBDIR}")
    gt_ids = sorted(gt_snapshots)
    print(f"Found {len(gt_ids)} GT snapshot(s): {gt_ids}")

    # --- weather conditions
    weather_conditions = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_dir() and p.name != GT_SUBDIR and (p / FINAL_SUBDIR).is_dir()
    )
    print(f"Found {len(weather_conditions)} weather condition(s): {weather_conditions}")

    clip_scorer = None if args.no_clip else ClipScorer()

    results = []

    for weather in weather_conditions:
        weather_dir = run_dir / weather
        finals = find_snapshots(weather_dir / FINAL_SUBDIR, "Final")
        pairs = find_image_pairs(weather_dir / IMAGES_SUBDIR)
        image_pairs = match_image_pairs(finals, pairs)
        print(f"\n[{weather}] {len(finals)} Final snapshot(s), {len(pairs)} Start/End pair(s)")

        for n in sorted(finals):
            final_time, final_dir = finals[n]
            gt_id = gt_ids[(n - 1) % len(gt_ids)]
            gt_dir = gt_snapshots[gt_id][1]
            start_path, end_path = image_pairs.get(n, (None, None))

            print(f"\n--- {weather} | Final{n} -> GT{gt_id}")

            trial_dir = results_dir / weather / f"trial{n}_GT{gt_id}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            title = f"{weather} | trial {n} | GT{gt_id} | {final_time:%Y-%m-%d %H:%M:%S}"

            gt_meta, final_meta = rt.load_meta(gt_dir), rt.load_meta(final_dir)

            # --- CLIP (GT vs Start and End) + image panel
            sims, panel_file = None, None
            if start_path is None:
                print("    WARNING: no Start/End images for this trial, skipping CLIP")
            elif clip_scorer is not None:
                gt_rgb = gt_dir / "rgb.png"
                sims = {"start": clip_scorer.similarity(gt_rgb, start_path),
                        "end": clip_scorer.similarity(gt_rgb, end_path)}
                print(f"    CLIP  start {sims['start']:.4f}   end {sims['end']:.4f}   "
                      f"improvement {sims['end'] - sims['start']:+.4f}")

                panel_file = trial_dir / "images.png"
                plot_panel(gt_dir, start_path, end_path, sims, panel_file, title)

            # --- pose error from metadata
            position_error = meta_pose_error(gt_meta, final_meta)
            print(f"    position error {position_error['norm']:.1f} cm")

            # --- does the translation error compensate the rotation error?
            # Rotation error -> pixel shift (K, R) -> equivalent camera translation
            # (depth), added to the actual translation error. Done with the GT depth
            # map and, as a check, with the Final depth map.
            compensation, figures = None, None
            if not args.no_depth:
                pc_gt = rt.position_compensation(gt_dir, final_dir)
                pc_final = rt.position_compensation(final_dir, gt_dir)
                rt_gt = rt.analyze(gt_dir, final_dir, trial_dir,
                                   max_points=args.max_points, verbose=args.verbose)
                compensation = compensation_report(pc_gt, pc_final, rt_gt)
                figures = rt_gt["figures"]

                lat = compensation["lateral_cm"]
                print(f"    rotation {compensation['rotation_error_deg']:.3f} deg = "
                      f"{compensation['rotation_pixel_shift_px']['median']:.2f} px = "
                      f"{lat['rotation_equivalent']:.1f} cm   translation "
                      f"{lat['translation_error']:.1f} cm (lateral)   -> visible "
                      f"{lat['visible_error']:.1f} cm = {compensation['pixel_shift_px']['both']:.2f} px")
                note = ""
                if not compensation["final_depth_check"]["agrees"]:
                    note = f"  (Final depth map says: {compensation['final_depth_check']['verdict']})"
                print(f"    -> {compensation['verdict']}{note}")

            # --- motion log plots (when the Final folder has one)
            motion = None
            gt_log, trial_log = gt_dir / "MotionLog.csv", final_dir / "MotionLog.csv"
            if gt_log.is_file() and trial_log.is_file():
                trans_plot = trial_dir / "comparison_plot_Trans.png"
                rot_plot = trial_dir / "comparison_plot_Rot.png"
                trial_df = load_motion_log(trial_log)
                plot_pose_comparison(load_motion_log(gt_log), trial_df,
                                     trans_plot, rot_plot, title)
                motion = {
                    "gt_motion_log": str(gt_log),
                    "motion_log": str(trial_log),
                    "duration_s": round(float(trial_df["time"].iloc[-1]), 3),
                    "plots": {"translation": str(trans_plot), "rotation": str(rot_plot)},
                }

            results.append({
                "weather": weather,
                "trial": n,
                "gt_id": gt_id,
                "final_timestamp": f"{final_time:%Y-%m-%d %H:%M:%S}",
                "ground_truth_dir": str(gt_dir),
                "final_dir": str(final_dir),
                "start_image": str(start_path) if start_path else None,
                "end_image": str(end_path) if end_path else None,
                "clip_similarity_to_gt": None if sims is None else {
                    "start": round(sims["start"], 6),
                    "end": round(sims["end"], 6),
                    "improvement": round(sims["end"] - sims["start"], 6),
                },
                "final_position_error_cm": position_error,
                "rotation_translation_compensation": compensation,
                "motion": motion,
                "images_plot": str(panel_file) if panel_file else None,
                "depth_figures": figures,
            })

    # --- YAML
    def summarize(rows):
        clip_rows = [r["clip_similarity_to_gt"] for r in rows if r["clip_similarity_to_gt"]]
        comp = [r["rotation_translation_compensation"] for r in rows
                if r["rotation_translation_compensation"]]
        verdicts = [c["verdict"] for c in comp]

        return {
            "number_of_trials": len(rows),
            "mean_start_similarity": mean([c["start"] for c in clip_rows]),
            "mean_end_similarity": mean([c["end"] for c in clip_rows]),
            "mean_improvement": mean([c["improvement"] for c in clip_rows]),
            "mean_final_position_error_cm": mean([r["final_position_error_cm"]["norm"] for r in rows]),
            "mean_rotation_error_deg": mean([c["rotation_error_deg"] for c in comp]),
            "compensation": {
                "trials_compensating": verdicts.count("compensating"),
                "trials_overcompensating": verdicts.count("overcompensating"),
                "trials_not_compensating": verdicts.count("not_compensating"),
                "trials_rotation_negligible": verdicts.count("rotation_negligible"),
                "trials_where_depth_maps_disagree": sum(
                    not c["final_depth_check"]["agrees"] for c in comp),
                "mean_rotation_pixel_shift_px": mean(
                    [c["rotation_pixel_shift_px"]["median"] for c in comp]),
                "mean_lateral_translation_error_cm": mean(
                    [c["lateral_cm"]["translation_error"] for c in comp]),
                "mean_lateral_rotation_equivalent_cm": mean(
                    [c["lateral_cm"]["rotation_equivalent"] for c in comp]),
                "mean_visible_error_cm": mean([c["lateral_cm"]["visible_error"] for c in comp]),
                "mean_pixel_shift_px": mean([c["pixel_shift_px"]["both"] for c in comp]),
                "mean_pixel_shift_in_clip_input_px": mean(
                    [c["pixel_shift_px"]["both_in_clip_input"] for c in comp]),
                "compensating_trials": [
                    r["trial"] for r in rows if r["rotation_translation_compensation"]
                    and r["rotation_translation_compensation"]["is_compensating"]],
            },
        }

    output = {
        "clip_model": None if args.no_clip else CLIP_MODEL,
        "metric": "cosine_similarity (GT rgb vs Start / End)",
        "compensation_method": {
            "steps": [
                "1. rotation error R (from the two meta.json) -> pixel shift of every pixel "
                "with the intrinsics only: u' = K R^T K^-1 u",
                "2. pixel shift -> position: each pixel's 3D point P from depth_planar_cm.f32; "
                "the camera translation that moves P by the same pixels is t_rot = P - R^T P "
                "(about depth * angle)",
                "3. visible error = part of (translation error + t_rot) perpendicular to the "
                "viewing ray; if it is smaller than the translation error alone, the "
                "translation is compensating the rotation",
                "4. visible error -> pixels (exact projection with the full End pose) and "
                "-> CLIP input pixels (image short side scaled to 224 px)",
            ],
            "vectors_frame": "GT camera: right / down / forward, cm",
            "depth_maps": "GT depth = main result; Final depth = final_depth_check",
            "world_xyz_check": ("depth_over_world_xyz_distance should be 1.0; exports made before "
                                "the camera-scale fix in GTSnapshotComponent.cpp give 10.0"),
            "verdict_rule": {
                "rotation_negligible": f"rotation pixel shift < {ROT_NEGLIGIBLE_PX} px",
                "compensating": (f"visible error < {COMPENSATING_RATIO} x both lateral translation "
                                 "error and lateral rotation-equivalent translation"),
                "overcompensating": "visible error < lateral translation error only",
                "not_compensating": "visible error >= lateral translation error",
            },
        },
        "weather_conditions": weather_conditions,
        "number_of_trials": len(results),
        "summary": {
            "overall": summarize(results),
            "by_weather": {w: summarize([r for r in results if r["weather"] == w])
                           for w in weather_conditions},
        },
        "trials": results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "similarity_results.yaml"
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)

    print("\n" + "=" * 60)
    print(f"Finished: {len(results)} trial(s)")
    print(f"Results written to {output_file}")
    print(f"Per-trial figures in {results_dir}/<weather>/trial<n>_GT<id>/")


if __name__ == "__main__":
    main()
