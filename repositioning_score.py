#!/usr/bin/env python3
"""
Two-stage similarity of START and FINAL frames against a TARGET frame,
returned as scores in [0, 1] (also printed as %).

Stage 1 - global appearance:
    DINOv2 CLS embedding, cosine similarity (clipped to [0, 1]).
Stage 2 - local geometry:
    SuperPoint + LightGlue matches, MAGSAC fundamental-matrix RANSAC.
      overlap   = inliers / min(#kpts target, #kpts query)
                  -> how much of the scene is re-observed
      alignment = 1 - median(normalised displacement of inliers) / d_max
                  -> how close the viewpoint is (0 at d_max of image size)

combined = weighted mean of (global, overlap, alignment).
If fewer than --min-inliers survive RANSAC, overlap and alignment are 0.

Install:
    pip install torch torchvision opencv-python numpy
    pip install git+https://github.com/cvg/LightGlue.git

Usage:
    python repositioning_score.py --target target.png --start start.png --final final.png
    python repositioning_score.py --target t.png --start s.png --final f.png --viz out/ --json scores.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TwoStageScorer:
    def __init__(self, device, dino="dinov2_vitb14", dino_size=448, max_kpts=2048,
                 resize=None, ransac_px=1.5, min_inliers=15, d_max=0.25,
                 weights=(0.2, 0.4, 0.4), dino_repo=None):
        self.dev = device
        self.dino_size = dino_size
        self.resize = resize
        self.ransac_px = ransac_px
        self.min_inliers = min_inliers
        self.d_max = d_max
        self.w = weights
        if dino_repo:  # local clone of facebookresearch/dinov2 (offline / behind a proxy)
            self.dino = torch.hub.load(dino_repo, dino, source="local")
        else:
            self.dino = torch.hub.load("facebookresearch/dinov2", dino)
        self.dino = self.dino.eval().to(device)
        self.extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        self._cache = {}

    @torch.no_grad()
    def _prepare(self, path):
        if path in self._cache:
            return self._cache[path]
        img = load_image(path)  # 3xHxW, float in [0, 1]
        H, W = img.shape[-2:]
        # Stage 1: global descriptor
        x = F.interpolate(img[None], size=(self.dino_size, self.dino_size),
                          mode="bilinear", align_corners=False)[0]
        x = ((x - MEAN) / STD)[None].to(self.dev)
        g = F.normalize(self.dino(x), dim=-1)[0]
        # Stage 2: local features (keypoints returned in original pixel coords;
        # resize=None detects them at full resolution)
        feats = self.extractor.extract(img.to(self.dev), resize=self.resize)
        entry = {"img": img, "size": np.array([W, H], dtype=np.float64), "g": g, "feats": feats}
        self._cache[path] = entry
        return entry

    @torch.no_grad()
    def score(self, target_path, query_path, exclude=None):
        """
        exclude: optional list of (x0, y0, x1, y1) boxes in normalised image coordinates (0..1).
        Matches with a keypoint inside any box in either image are dropped before RANSAC, e.g. a
        burned-in on-screen overlay that would otherwise match itself with zero displacement.
        """
        t = self._prepare(target_path)
        q = self._prepare(query_path)

        global_sim = float(torch.clamp(t["g"] @ q["g"], 0.0, 1.0))

        out = self.matcher({"image0": t["feats"], "image1": q["feats"]})
        f0, f1, m = rbd(t["feats"]), rbd(q["feats"]), rbd(out)
        idx = m["matches"]
        p0 = f0["keypoints"][idx[:, 0]].cpu().numpy().astype(np.float64)
        p1 = f1["keypoints"][idx[:, 1]].cpu().numpy().astype(np.float64)
        n0, n1 = len(f0["keypoints"]), len(f1["keypoints"])
        if exclude:
            def outside(p, size):
                u = p / size
                inside = np.zeros(len(p), dtype=bool)
                for x0, y0, x1, y1 in exclude:
                    inside |= (u[:, 0] >= x0) & (u[:, 0] <= x1) & (u[:, 1] >= y0) & (u[:, 1] <= y1)
                return ~inside
            keep = outside(p0, t["size"]) & outside(p1, q["size"])
            p0, p1 = p0[keep], p1[keep]

        inl = np.zeros(len(p0), dtype=bool)
        if len(p0) >= 8:
            _, mask = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC,
                                             self.ransac_px, 0.999, 10000)
            if mask is not None:
                inl = mask.ravel().astype(bool)
        n_inl = int(inl.sum())

        if n_inl >= self.min_inliers:
            overlap = min(1.0, n_inl / max(1, min(n0, n1)))
            a = p0[inl] / t["size"]  # normalise by each image's own size
            b = p1[inl] / q["size"]
            disp = float(np.median(np.linalg.norm(a - b, axis=1)))
            alignment = max(0.0, 1.0 - disp / self.d_max)
        else:
            overlap, alignment, disp = 0.0, 0.0, float("nan")

        wg, wo, wa = self.w
        combined = (wg * global_sim + wo * overlap + wa * alignment) / (wg + wo + wa)

        result = {
            "combined": combined,
            "global": global_sim,
            "overlap": overlap,
            "alignment": alignment,
            "matches": int(len(p0)),
            "inliers": n_inl,
            "median_disp_norm": disp,
            "kpts_target": n0,
            "kpts_query": n1,
        }
        return result, (p0[inl], p1[inl], t["img"], q["img"])


def draw_matches(pts_t, pts_q, img_t, img_q, out_path, max_lines=300):
    def to_bgr(img):
        a = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)

    A, B = to_bgr(img_t), to_bgr(img_q)
    h = max(A.shape[0], B.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    canvas = np.hstack([pad(A), pad(B)])
    off = A.shape[1]
    sel = np.linspace(0, len(pts_t) - 1, min(max_lines, len(pts_t))).astype(int) if len(pts_t) else []
    for i in sel:
        x0, y0 = pts_t[i]
        x1, y1 = pts_q[i]
        cv2.line(canvas, (int(x0), int(y0)), (int(x1) + off, int(y1)), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(canvas, "QUERY", (off + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dino", default="dinov2_vitb14",
                    help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14")
    ap.add_argument("--dino-repo", help="path to a local clone of facebookresearch/dinov2")
    ap.add_argument("--max-kpts", type=int, default=2048)
    ap.add_argument("--resize", type=int, default=None,
                    help="long side for SuperPoint (default: full resolution, no resizing)")
    ap.add_argument("--ransac-px", type=float, default=1.5)
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--d-max", type=float, default=0.25,
                    help="normalised displacement at which alignment hits 0")
    ap.add_argument("--weights", type=float, nargs=3, default=[0.2, 0.4, 0.4],
                    metavar=("GLOBAL", "OVERLAP", "ALIGN"))
    ap.add_argument("--self-check", action="store_true",
                    help="also score target vs itself (should be ~100%%)")
    ap.add_argument("--viz", help="directory to save match visualisations")
    ap.add_argument("--json", help="path to save scores as JSON")
    args = ap.parse_args()

    scorer = TwoStageScorer(args.device, args.dino, max_kpts=args.max_kpts, resize=args.resize,
                            ransac_px=args.ransac_px, min_inliers=args.min_inliers,
                            d_max=args.d_max, weights=tuple(args.weights),
                            dino_repo=args.dino_repo)

    pairs = {"start": args.start, "final": args.final}
    if args.self_check:
        pairs["target(self)"] = args.target

    results = {}
    for name, path in pairs.items():
        res, vis = scorer.score(args.target, path)
        results[name] = res
        if args.viz:
            Path(args.viz).mkdir(parents=True, exist_ok=True)
            draw_matches(*vis, Path(args.viz) / f"matches_target_vs_{name.split('(')[0]}.png")

    hdr = f"{'pair':<14}{'combined':>10}{'global':>9}{'overlap':>9}{'align':>8}{'inliers':>9}{'disp':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<14}{r['combined']*100:>9.1f}%{r['global']*100:>8.1f}%"
              f"{r['overlap']*100:>8.1f}%{r['alignment']*100:>7.1f}%"
              f"{r['inliers']:>9d}{r['median_disp_norm']:>8.3f}")
    gain = results["final"]["combined"] - results["start"]["combined"]
    print(f"\nImprovement (final - start): {gain*100:+.1f} percentage points")

    if args.json:
        results["improvement"] = gain
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
