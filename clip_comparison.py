import re
from pathlib import Path

import torch
import clip
import yaml
from PIL import Image
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

GT_DIR = Path("data/imagesGT4")
RESULTS_DIR = Path("data/results4_0")
OUTPUT_FILE = Path("data/results4_0/similarity_results.yaml")

WEATHER_FOLDERS = ["rain", "sunny", "snow"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Filename prefix of the image captured BEFORE repositioning starts, e.g.
# "Start1_Capture_20260911_135900.png". Stored next to the Final images in
# RESULTS_DIR/<weather>/. Images are matched to a GT by their integer ID.
START_PREFIX = "Start"


# ============================================================
# Device and CLIP model
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


# ============================================================
# Helper functions
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


def get_final_id(filename):
    """
    Extract the integer ID from:
        Final1_Capture_20260827_141931.png

    Returns:
        1
    """

    match = re.match(r"^Final(\d+)_Capture_", filename)

    if match is None:
        return None

    return int(match.group(1))


def get_prefixed_id(filename, prefix):
    """
    Extract the integer ID from '<prefix><id>_Capture_...' (None if no match).
    """

    match = re.match(rf"^{re.escape(prefix)}(\d+)_Capture_", filename)

    if match is None:
        return None

    return int(match.group(1))


def same_file(path1, path2):
    """
    True if both files are byte-identical (catches a GT copied in as 'start').
    """

    return path1.read_bytes() == path2.read_bytes()


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


# ============================================================
# Find ground-truth images
# ============================================================

print("\nScanning ground-truth images...")

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


print(f"Found {len(gt_images)} GT images.")


# ============================================================
# Find result images
# ============================================================

print("\nScanning result images...")

final_images = {}
start_images = {}

for weather in WEATHER_FOLDERS:

    weather_dir = RESULTS_DIR / weather

    if not weather_dir.exists():
        print(f"WARNING: Weather directory does not exist: {weather_dir}")
        continue

    for path in weather_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        start_id = get_prefixed_id(path.name, START_PREFIX)

        if start_id is not None:
            start_images[start_id] = path
            continue

        image_id = get_final_id(path.name)

        if image_id is None:
            # GT copies etc. living in the results folder are not results.
            if get_gt_id(path.name) is None:
                print(f"WARNING: Could not parse Final ID: {path}")
            continue

        if image_id in final_images:
            raise RuntimeError(
                f"Duplicate Final ID {image_id}:\n"
                f"  {final_images[image_id]['path']}\n"
                f"  {path}"
            )

        final_images[image_id] = {
            "path": path,
            "weather": weather,
        }


print(f"Found {len(final_images)} result images "
      f"and {len(start_images)} pre-repositioning ('{START_PREFIX}') images.")


# ============================================================
# Check matching IDs
# ============================================================

gt_ids = set(gt_images.keys())
final_ids = set(final_images.keys())

missing_results = sorted(gt_ids - final_ids)
missing_gt = sorted(final_ids - gt_ids)

if missing_results:
    print("\nWARNING: GT images without matching Final images:")
    for image_id in missing_results:
        print(f"  ID {image_id}")

if missing_gt:
    print("\nWARNING: Final images without matching GT images:")
    for image_id in missing_gt:
        print(f"  ID {image_id}")


matching_ids = sorted(gt_ids & final_ids)

print(f"\nMatched image pairs: {len(matching_ids)}")


# ============================================================
# Calculate similarities
# ============================================================

results = []

print("\nCalculating CLIP similarities...\n")

for index, image_id in enumerate(matching_ids, start=1):

    gt_path = gt_images[image_id]

    final_info = final_images[image_id]
    final_path = final_info["path"]
    weather = final_info["weather"]

    print(
        f"[{index}/{len(matching_ids)}] "
        f"ID {image_id} | {weather}"
    )

    final_sim = scene_similarity(gt_path, final_path)

    pair = {
        "id": image_id,

        "ground_truth": {
            "filename": gt_path.name,
            "path": str(gt_path),
        },

        "start": None,

        "final": {
            "filename": final_path.name,
            "path": str(final_path),
            "weather": weather,
            "similarity_to_gt": round(final_sim, 6),
        },
    }

    start_path = start_images.get(image_id)

    if start_path is None:
        print(f"    WARNING: no '{START_PREFIX}{image_id}_Capture_*' image")
    else:
        if same_file(gt_path, start_path):
            print("    WARNING: start image is identical to the GT image")

        start_sim = scene_similarity(gt_path, start_path)

        pair["start"] = {
            "filename": start_path.name,
            "path": str(start_path),
            "similarity_to_gt": round(start_sim, 6),
        }

        # > 0 means repositioning made the view more similar to the GT
        pair["improvement"] = round(final_sim - start_sim, 6)

        print(f"    Start similarity: {start_sim:.6f}")
        print(f"    Improvement:      {final_sim - start_sim:+.6f}")

    print(f"    Final similarity: {final_sim:.6f}")

    results.append(pair)


def mean(values):
    return round(sum(values) / len(values), 6) if values else None


# ============================================================
# Write YAML
# ============================================================

output = {
    "model": "ViT-B/32",
    "metric": "cosine_similarity",
    "source_directory": str(GT_DIR),
    "target_directory": str(RESULTS_DIR),
    "weather_conditions": WEATHER_FOLDERS,
    "number_of_pairs": len(results),
    "summary": {
        "mean_start_similarity": mean(
            [r["start"]["similarity_to_gt"] for r in results if r["start"]]),
        "mean_final_similarity": mean(
            [r["final"]["similarity_to_gt"] for r in results]),
        "mean_improvement": mean(
            [r["improvement"] for r in results if "improvement" in r]),
    },
    "pairs": results,
}


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

print(f"GT images:       {len(gt_images)}")
print(f"Final images:    {len(final_images)}")
print(f"Matched pairs:   {len(results)}")
print(f"Missing results: {len(missing_results)}")
print(f"Missing GT:      {len(missing_gt)}")

print(f"\nResults written to:")
print(f"  {OUTPUT_FILE}")