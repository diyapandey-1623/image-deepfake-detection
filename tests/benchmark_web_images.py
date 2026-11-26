"""
tests/benchmark_web_images.py
-----------------------------
Evaluates the active Hugging Face model (or any --model_id) on real-world web
test images in tests/web_benchmark/{real,fake} using the backend's exact
EXIF-transpose + 1024px-JPEG-normalization + MTCNN-crop pipeline.
"""

import argparse
import io
import sys
from pathlib import Path
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facenet_pytorch import MTCNN
from backend.config import FACE_MARGIN_PERCENT, MODEL_ID
from backend.model.checks import detect_and_crop_faces
from backend.model.inference import HuggingFaceModel


def prepare_image(path: Path) -> Image.Image:
    raw = Image.open(path)
    pil_image = ImageOps.exif_transpose(raw).convert("RGB")
    if max(pil_image.size) > 1024:
        pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        pil_image = Image.open(buf).convert("RGB")
    return pil_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a Hugging Face deepfake model on web test images.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Hugging Face model ID or local path")
    args = parser.parse_args()

    bench_dir = PROJECT_ROOT / "tests" / "web_benchmark"
    real_files = sorted((bench_dir / "real").glob("*.jpg"))
    fake_files = sorted((bench_dir / "fake").glob("*.jpg"))

    mtcnn = MTCNN(keep_all=True, device="cpu")
    model = HuggingFaceModel(model_id=args.model_id, device="cpu")

    correct = 0
    total = 0

    for expected, files in (("real", real_files), ("fake", fake_files)):
        for path in files:
            img = prepare_image(path)
            crops, detected = detect_and_crop_faces(img, mtcnn, margin_percent=FACE_MARGIN_PERCENT)
            target = crops[0][0] if (detected and crops) else img
            res = model.predict(target)
            ok = res["verdict"] == expected
            correct += int(ok)
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] expected={expected:4} | got={res['verdict']:4} (p_fake={res['fake_probability']:.4f}) | {path.name}")

    print(f"\nAccuracy: {correct}/{total} ({(correct / max(1, total)) * 100:.1f}%) on {args.model_id}")


if __name__ == "__main__":
    main()
