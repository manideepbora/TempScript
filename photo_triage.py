#!/usr/bin/env python3
"""
photo_triage.py — 100% offline photo deduplication and quality triage.

Nothing is uploaded anywhere. The only network access is a one-time download
of the CLIP weights (~350 MB) on first run; after that you can stay offline.

Stages
  1. phash    exact + near-duplicate detection (resizes, re-saves, crops)
  2. clip     semantic clustering (burst shots, same scene different frame)
  3. quality  blur (Laplacian variance) + exposure + resolution scoring

Nothing is deleted. Rejects are MOVED into a review folder (or just listed,
in the default dry-run mode) so you can eyeball them before removing anything.

Install:
    python3 -m venv venv && source venv/bin/activate
    pip install torch torchvision open_clip_torch pillow pillow-heif \
                imagehash opencv-python numpy tqdm

Usage:
    python photo_triage.py ~/Pictures                     # dry run, writes report.csv
    python photo_triage.py ~/Pictures --apply             # actually move rejects
    python photo_triage.py ~/Pictures --no-clip           # skip the slow stage
    python photo_triage.py ~/Pictures --blur-threshold 60 --clip-threshold 0.95
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("note: pillow-heif not installed, .HEIC files will be skipped")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp",
              ".tif", ".tiff", ".bmp", ".gif"}


# ----------------------------------------------------------------- utilities

class UnionFind:
    """Groups items that are transitively similar to each other."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return [g for g in out.values() if len(g) > 1]


def find_images(root):
    files = [p for p in Path(root).rglob("*")
             if p.is_file()
             and p.suffix.lower() in EXTENSIONS
             and not p.name.startswith(".")
             and "_photo_review" not in p.parts]
    return sorted(files)


def load_rgb(path, max_side=None):
    img = Image.open(path)
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


# ------------------------------------------------------------ stage 3: quality

def quality_metrics(path):
    """Cheap, reliable, no model needed. Returns None if unreadable."""
    try:
        img = load_rgb(path, max_side=1024)
    except Exception:
        return None

    arr = np.asarray(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Blur: variance of the Laplacian. Low = few sharp edges = soft/blurry.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness = float(gray.mean())
    contrast = float(gray.std())

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum()
    blown = float(hist[250:].sum())    # clipped highlights
    crushed = float(hist[:6].sum())    # crushed shadows

    try:
        with Image.open(path) as probe:
            w, h = probe.size
    except Exception:
        h, w = gray.shape

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "blown": blown,
        "crushed": crushed,
        "megapixels": (w * h) / 1_000_000,
    }


def quality_flags(m, blur_threshold):
    """Human-readable reasons this photo looks bad. Empty list = fine."""
    flags = []
    if m["sharpness"] < blur_threshold:
        flags.append("blurry")
    if m["brightness"] < 35 or m["crushed"] > 0.55:
        flags.append("underexposed")
    if m["brightness"] > 225 or m["blown"] > 0.40:
        flags.append("overexposed")
    if m["contrast"] < 12:
        flags.append("flat")
    return flags


def keeper_score(m):
    """Used to pick the best photo out of a duplicate group."""
    return (np.log1p(m["sharpness"]) * 2.0
            + min(m["megapixels"], 24) * 0.15
            + m["contrast"] * 0.02
            - (m["blown"] + m["crushed"]) * 3.0)


# -------------------------------------------------------------- stage 1: phash

def compute_hashes(paths):
    hashes, ok = [], []
    for p in tqdm(paths, desc="1/3 hashing", unit="img"):
        try:
            h = imagehash.phash(load_rgb(p, max_side=256), hash_size=8)
            hashes.append(h.hash.flatten())
            ok.append(p)
        except Exception:
            continue
    return ok, (np.array(hashes, dtype=np.uint8) if hashes else np.empty((0, 64), np.uint8))


def hash_groups(bits, max_distance):
    """Pairwise Hamming distance via matrix multiply. Fine up to ~30k images."""
    n = len(bits)
    if n < 2:
        return []
    b = bits.astype(np.float32)
    inv = 1.0 - b
    # matching bits = b·bᵀ + inv·invᵀ  →  distance = 64 - matches
    matches = b @ b.T + inv @ inv.T
    dist = bits.shape[1] - matches
    np.fill_diagonal(dist, 999)

    uf = UnionFind(n)
    for i, j in zip(*np.where(dist <= max_distance)):
        if i < j:
            uf.union(int(i), int(j))
    return uf.groups()


# --------------------------------------------------------------- stage 2: CLIP

def compute_embeddings(paths, batch_size=32):
    import torch
    import open_clip

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"    CLIP device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()

    vectors, ok = [], []
    for start in tqdm(range(0, len(paths), batch_size),
                      desc="2/3 embedding", unit="batch"):
        chunk = paths[start:start + batch_size]
        tensors, valid = [], []
        for p in chunk:
            try:
                tensors.append(preprocess(load_rgb(p, max_side=512)))
                valid.append(p)
            except Exception:
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats /= feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())
        ok.extend(valid)

    if not vectors:
        return [], np.empty((0, 512), np.float32)
    return ok, np.vstack(vectors).astype(np.float32)


def embedding_groups(vecs, threshold, block=1024):
    n = len(vecs)
    if n < 2:
        return []
    uf = UnionFind(n)
    for start in range(0, n, block):
        sim = vecs[start:start + block] @ vecs.T
        for local_i, row in enumerate(sim):
            i = start + local_i
            for j in np.where(row >= threshold)[0]:
                if i < j:
                    uf.union(i, int(j))
    return uf.groups()


# ---------------------------------------------------------------------- output

def relocate(path, root, review_root, bucket):
    dest_dir = review_root / bucket / path.parent.relative_to(root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), str(dest))
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder to scan (searched recursively)")
    ap.add_argument("--apply", action="store_true",
                    help="actually move rejects; default is dry run")
    ap.add_argument("--no-clip", action="store_true",
                    help="skip semantic clustering (much faster)")
    ap.add_argument("--hash-distance", type=int, default=5,
                    help="phash Hamming tolerance, 0-12 (default 5; higher = looser)")
    ap.add_argument("--clip-threshold", type=float, default=0.94,
                    help="cosine similarity for same-scene (default 0.94)")
    ap.add_argument("--blur-threshold", type=float, default=45.0,
                    help="Laplacian variance below this is 'blurry' (default 45)")
    args = ap.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    review_root = root / "_photo_review"

    paths = find_images(root)
    print(f"found {len(paths)} images under {root}\n")
    if not paths:
        return

    # Stage 1
    paths, bits = compute_hashes(paths)
    dup_groups = hash_groups(bits, args.hash_distance)
    print(f"    {len(dup_groups)} near-duplicate groups\n")

    # Stage 3 (run before pruning so we can pick the best of each group)
    metrics = {}
    for p in tqdm(paths, desc="3/3 quality", unit="img"):
        m = quality_metrics(p)
        if m:
            metrics[p] = m

    # Stage 2
    scene_groups = []
    if not args.no_clip:
        clip_paths, vecs = compute_embeddings(paths)
        raw = embedding_groups(vecs, args.clip_threshold)
        scene_groups = [[clip_paths[i] for i in g] for g in raw]
        print(f"    {len(scene_groups)} similar-scene groups\n")

    dup_groups = [[paths[i] for i in g] for g in dup_groups]

    # Decide the fate of each file
    verdict = {}   # path -> (bucket, reason, group_id)
    group_id = 0

    for kind, groups in (("duplicate", dup_groups), ("similar", scene_groups)):
        for group in groups:
            group = [p for p in group if p not in verdict]
            if len(group) < 2:
                continue
            group_id += 1
            best = max(group, key=lambda p: keeper_score(metrics[p])
                       if p in metrics else -1e9)
            for p in group:
                if p is not best:
                    verdict[p] = (kind, f"{kind} of {best.name}", group_id)

    for p, m in metrics.items():
        if p in verdict:
            continue
        flags = quality_flags(m, args.blur_threshold)
        if flags:
            verdict[p] = ("low_quality", ", ".join(flags), 0)

    # Report
    report = root / "photo_triage_report.csv"
    with open(report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "verdict", "reason", "group",
                    "sharpness", "brightness", "contrast", "megapixels"])
        for p in paths:
            bucket, reason, gid = verdict.get(p, ("keep", "", 0))
            m = metrics.get(p, {})
            w.writerow([
                p, bucket, reason, gid or "",
                f"{m.get('sharpness', 0):.1f}",
                f"{m.get('brightness', 0):.1f}",
                f"{m.get('contrast', 0):.1f}",
                f"{m.get('megapixels', 0):.1f}",
            ])

    counts = defaultdict(int)
    for bucket, _, _ in verdict.values():
        counts[bucket] += 1

    print(f"\nkeep:        {len(paths) - len(verdict)}")
    print(f"duplicate:   {counts['duplicate']}")
    print(f"similar:     {counts['similar']}")
    print(f"low_quality: {counts['low_quality']}")
    print(f"\nreport: {report}")

    if not args.apply:
        print("\nDry run — nothing moved. Review the CSV, then re-run with --apply.")
        return

    for p, (bucket, _, gid) in tqdm(verdict.items(), desc="moving", unit="file"):
        sub = f"{bucket}/group_{gid:04d}" if gid else bucket
        try:
            relocate(p, root, review_root, sub)
        except Exception as e:
            print(f"could not move {p}: {e}")
    print(f"\nrejects moved to {review_root} — delete that folder once you've checked it.")


if __name__ == "__main__":
    main()
