"""
scripts/make_sample.py
----------------------
Deterministic dataset sampler for deepfake detection benchmark.
Extracts a balanced subset from the Kaggle 140k Real and Fake Faces dataset
using fixed seed=42, and generates an cryptographic SHA256 manifest.csv.
"""

import os
import sys
import shutil
import random
import hashlib
import argparse
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 hex digest of a binary file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_raw_dataset(candidate_paths=None) -> Path:
    """Locates the uncompressed Kaggle 140k real and fake dataset root."""
    if candidate_paths is None:
        candidate_paths = [
            _PROJECT_ROOT / "data" / "raw",
            _PROJECT_ROOT / "data" / "real_vs_fake",
            _PROJECT_ROOT / "data" / "140k-real-and-fake-faces",
            _PROJECT_ROOT.parent / "data" / "raw",
            Path.home() / ".kaggle" / "datasets" / "xhlulu" / "140k-real-and-fake-faces"
        ]

    for p in candidate_paths:
        # Check standard splits or real/fake directories
        if (p / "train" / "real").is_dir() or (p / "real").is_dir() or (p / "real_vs_fake" / "train" / "real").is_dir():
            return p
    return None


def generate_manifest(small_dir: Path, manifest_path: Path):
    """Generates a CSV manifest with relative filepaths, class labels, and SHA256 hashes."""
    import csv

    records = []
    splits = ["train", "valid", "test"]
    classes = ["real", "fake"]

    for split in splits:
        for c in classes:
            folder = small_dir / split / c
            if not folder.is_dir():
                continue
            for f in sorted(folder.glob("*.*")):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    rel_p = str(f.relative_to(small_dir)).replace("\\", "/")
                    sha = compute_sha256(f)
                    records.append({
                        "split": split,
                        "class_name": c,
                        "filename": f.name,
                        "relative_path": rel_p,
                        "sha256": sha
                    })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "filename", "relative_path", "sha256"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[Manifest] Generated manifest with {len(records)} entries -> {manifest_path}")
    return records


def sample_dataset(source_dir: Path, target_dir: Path, seed: int = 42):
    """Samples exactly 1,350 balanced images (700 train, 150 valid, 500 test) using seed 42."""
    rng = random.Random(seed)

    # Resolve actual image root inside source
    if (source_dir / "real_vs_fake").is_dir():
        source_dir = source_dir / "real_vs_fake"

    split_quotas = {
        "train": 350,  # 350 real + 350 fake = 700
        "valid": 75,   # 75 real + 75 fake = 150
        "test": 250    # 250 real + 250 fake = 500
    }

    print(f"[Sampler] Sampling from {source_dir} -> {target_dir} (Seed: {seed})...")

    for split, count_per_class in split_quotas.items():
        for c in ["real", "fake"]:
            src_split_folder = source_dir / split / c
            # Fallback to monolithic folder if no sub-split
            if not src_split_folder.is_dir():
                src_split_folder = source_dir / c

            if not src_split_folder.is_dir():
                raise FileNotFoundError(f"Source image folder not found: {src_split_folder}")

            all_files = sorted([f for f in src_split_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
            if len(all_files) < count_per_class:
                raise ValueError(f"Insufficient images in {src_split_folder}: requested {count_per_class}, found {len(all_files)}")

            # Deterministic selection with seeded RNG
            chosen_files = rng.sample(all_files, count_per_class)

            dest_folder = target_dir / split / c
            dest_folder.mkdir(parents=True, exist_ok=True)

            for f in chosen_files:
                shutil.copy2(f, dest_folder / f.name)

    print("[Sampler] Sampling completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Deterministic dataset sampler and manifest builder.")
    parser.add_argument("--source_dir", type=str, default=None, help="Path to raw Kaggle 140k dataset")
    parser.add_argument("--target_dir", type=str, default=str(_PROJECT_ROOT / "data" / "small"), help="Output directory for data/small")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--manifest_only", action="store_true", help="Only rebuild manifest.csv from existing target_dir")
    args = parser.parse_args()

    target_path = Path(args.target_dir)

    if args.manifest_only or (target_path.is_dir() and any((target_path / "test").glob("*/*")) and not args.source_dir):
        print(f"[Mode] Existing sample found at {target_path}. Generating manifest.csv...")
        manifest_path = target_path / "manifest.csv"
        records = generate_manifest(target_path, manifest_path)
        # Also copy manifest to reference_results for git tracking
        ref_manifest = _PROJECT_ROOT / "reference_results" / "manifest.csv"
        ref_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, ref_manifest)
        print(f"[Manifest] Mirrored manifest to tracked location: {ref_manifest}")
        return

    # Sampling from raw dataset
    raw_dir = Path(args.source_dir) if args.source_dir else find_raw_dataset()
    if not raw_dir or not raw_dir.is_dir():
        print("[Error] Raw Kaggle dataset folder not found.")
        print("Please download the dataset from: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print("Extract it to data/raw/ or provide --source_dir /path/to/extracted/dataset")
        sys.exit(1)

    sample_dataset(raw_dir, target_path, seed=args.seed)
    manifest_path = target_path / "manifest.csv"
    generate_manifest(target_path, manifest_path)
    ref_manifest = _PROJECT_ROOT / "reference_results" / "manifest.csv"
    ref_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, ref_manifest)


if __name__ == "__main__":
    main()
