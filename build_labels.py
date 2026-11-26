"""
build_labels.py
----------------
Scans a dataset directory structure and creates labels.csv automatically.

Expected standard directory patterns supported:
1) Subfolders by class:
   data/raw/real/person_01/img1.jpg
   data/raw/fake/person_01_synthetic/img2.jpg
   or:
   data/raw/real/img1.jpg
   data/raw/fake/img2.jpg

2) Subfolders by subject:
   data/raw/subject_01/real/img1.jpg
   data/raw/subject_01/fake_faceswap/img2.jpg
   data/raw/subject_01/fake_diffusion/img3.jpg

CSV Output schema:
   path,label,subject_id,category
   - path: relative or absolute path to image file
   - label: 0 for REAL, 1 for FAKE
   - subject_id: identifier used for grouped/leak-free splitting (train/val/test)
   - category: free text descriptor (e.g., 'real_photo', 'faceswap', 'stylegan', 'diffusion')
"""

import os
import csv
import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_label_and_category(parts):
    """
    Infers label (0 for real, 1 for fake) and category name from directory path parts.
    Design rationale for viva:
    Robust keyword matching enables seamless ingestion of academic benchmarks
    (FaceForensics++, Celeb-DF, WildDeepfake, 100K-Faces) and custom user datasets.
    """
    lower_parts = [p.lower() for p in parts]
    
    # Check for fake / synthetic indicators
    is_fake = any(kw in p for p in lower_parts for kw in ["fake", "synthetic", "manipulated", "deepfake", "swap", "diffusion", "gan", "generated"])
    is_real = any(kw in p for p in lower_parts for kw in ["real", "genuine", "authentic", "original", "pristine"])
    
    if is_fake and not is_real:
        label = 1
    elif is_real and not is_fake:
        label = 0
    else:
        # Default heuristic: if 'fake' is anywhere in the filename or immediate folder
        label = 1 if any("fake" in p for p in lower_parts) else 0

    # Determine category
    category = "real_capture" if label == 0 else "synthetic_deepfake"
    for p in lower_parts:
        for cat_kw in ["faceswap", "deepfakes", "face2face", "neuraltextures", "diffusion", "stylegan", "midjourney"]:
            if cat_kw in p:
                category = cat_kw
                break
                
    return label, category


def infer_subject_id(file_path: Path, root_path: Path):
    """
    Extracts a subject identifier from directory hierarchy to prevent data leakage.
    Example: data/raw/person_01/real/img.jpg -> subject_id: person_01
    """
    rel_path = file_path.relative_to(root_path)
    parts = list(rel_path.parts)
    
    # If nested in a subject folder (e.g. subject_01/real/...), use that folder name
    for part in parts[:-1]:
        part_lower = part.lower()
        if part_lower not in ["real", "fake", "raw", "images", "dataset", "test", "train", "val"]:
            return part
            
    # Fallback to parent directory name or file stem base prefix
    return file_path.parent.name if file_path.parent != root_path else file_path.stem.split("_")[0]


def scan_and_build_labels(data_dir: str, output_csv: str):
    root = Path(data_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    rows = []
    print(f"Scanning {root} for image files...")

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel_parts = path.relative_to(root).parts
            label, category = infer_label_and_category(rel_parts)
            subject_id = infer_subject_id(path, root)
            
            # Store path relative to project root or data_dir
            rows.append({
                "path": str(path.as_posix()),
                "label": label,
                "subject_id": subject_id,
                "category": category
            })

    # Sort for deterministic output
    rows.sort(key=lambda x: x["path"])

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "subject_id", "category"])
        writer.writeheader()
        writer.writerows(rows)

    num_real = sum(1 for r in rows if r["label"] == 0)
    num_fake = sum(1 for r in rows if r["label"] == 1)
    subjects = len(set(r["subject_id"] for r in rows))

    print(f"[Done] Indexed {len(rows)} images -> {output_path}")
    print(f"  Real: {num_real} | Fake: {num_fake} | Unique subjects: {subjects}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan folder tree and create labels.csv")
    parser.add_argument("--data_dir", type=str, default="data/small", help="Path to image directory")
    parser.add_argument("--output_csv", type=str, default="data/labels.csv", help="Output path for labels.csv")
    args = parser.parse_args()
    
    scan_and_build_labels(args.data_dir, args.output_csv)
