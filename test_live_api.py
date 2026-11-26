"""Manual smoke-test script for a running FastAPI backend at http://127.0.0.1:8000."""

import os
from pathlib import Path
import requests


def main() -> None:
    project_root = Path(__file__).resolve().parent
    test_dir = project_root / "data" / "small" / "test"
    url = os.getenv("API_URL", "http://127.0.0.1:8000") + "/api/analyze"

    for label in ("real", "fake"):
        files = sorted((test_dir / label).iterdir())[:3]
        for fpath in files:
            with open(fpath, "rb") as f:
                res = requests.post(url, files={"file": (fpath.name, f, "image/jpeg")}, timeout=30)
            data = res.json()
            print(
                f"[{label.upper():4}] {fpath.name}: verdict={data.get('verdict')} "
                f"| fake_prob={data.get('fake_probability'):.4f} | regions={len(data.get('regional_influence', []))}"
            )


if __name__ == "__main__":
    main()
