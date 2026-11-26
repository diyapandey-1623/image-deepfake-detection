"""
scripts/verify_data.py
----------------------
Integrity verification script for dataset samples.
Validates all image files against data/small/manifest.csv using SHA256 hashes.
"""

import sys
import csv
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


def verify_dataset(manifest_path: Path, data_dir: Path, split_filter: str = None) -> bool:
    """Verifies that all files in manifest exist in data_dir and match SHA256 hashes."""
    if not manifest_path.is_file():
        print(f"[Error] Manifest file not found: {manifest_path}")
        return False

    if not data_dir.is_dir():
        print(f"[Error] Data directory not found: {data_dir}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if split_filter:
        reader = [r for r in reader if r.get("split") == split_filter]

    total = len(reader)
    passed = 0
    missing = []
    mismatched = []

    print(f"\n[Verification] Verifying {total} image files in {data_dir} against {manifest_path.name}...")

    for row in reader:
        rel_p = row["relative_path"]
        expected_sha = row["sha256"]
        full_p = data_dir / rel_p

        if not full_p.is_file():
            missing.append(rel_p)
            continue

        actual_sha = compute_sha256(full_p)
        if actual_sha.lower() != expected_sha.lower():
            mismatched.append((rel_p, expected_sha, actual_sha))
        else:
            passed += 1

    print("\n" + "=" * 60)
    print("               DATASET INTEGRITY REPORT               ")
    print("=" * 60)
    print(f"Total Manifest Records:  {total}")
    print(f"Successfully Verified:   {passed} / {total} ({(passed/total)*100:.1f}%)")
    print(f"Missing Files:           {len(missing)}")
    print(f"SHA256 Mismatches:       {len(mismatched)}")
    print("-" * 60)

    if missing:
        print("\nMissing files (first 5):")
        for m in missing[:5]:
            print(f"  - {m}")

    if mismatched:
        print("\nMismatched files (first 5):")
        for m, exp, act in mismatched[:5]:
            print(f"  - {m} (expected: {exp[:10]}..., got: {act[:10]}...)")

    is_clean = len(missing) == 0 and len(mismatched) == 0
    status_str = "PASS - DATASET INTEGRITY VERIFIED" if is_clean else "FAIL - INTEGRITY ISSUES DETECTED"
    print("\nOVERALL STATUS: " + status_str)
    print("=" * 60 + "\n")

    return is_clean


def main():
    parser = argparse.ArgumentParser(description="Verify dataset samples against SHA256 manifest.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.csv")
    parser.add_argument("--data_dir", type=str, default=str(_PROJECT_ROOT / "data" / "small"), help="Path to data/small")
    parser.add_argument("--split", type=str, default=None, help="Filter verification to specific split (train/valid/test)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    if not manifest_path or not manifest_path.exists():
        candidates = [
            Path(args.data_dir) / "manifest.csv",
            _PROJECT_ROOT / "reference_results" / "manifest.csv"
        ]
        for c in candidates:
            if c.is_file():
                manifest_path = c
                break

    if not manifest_path:
        print("[Error] Could not find manifest.csv. Run `python scripts/make_sample.py` first.")
        sys.exit(1)

    success = verify_dataset(manifest_path, Path(args.data_dir), split_filter=args.split)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
