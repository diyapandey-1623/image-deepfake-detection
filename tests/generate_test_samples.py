"""
generate_test_samples.py
-------------------------
Generates sample synthetic portraits for offline tests and smoke tests.
"""

from pathlib import Path
import cv2
import numpy as np

from backend.model.samples import create_synthetic_portrait


def generate_sample_dataset(output_dir: str = "data/raw", num_samples: int = 12):
    """
    Generates a starter dataset with balanced real/fake subjects for immediate testing.
    """
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} sample images into {base_path}...")
    np.random.seed(42)

    for i in range(num_samples):
        is_fake = (i % 2 != 0)
        subject_id = f"subject_{i // 2 + 1:02d}"
        kind = "fake_gan" if is_fake else "real"
        
        folder = base_path / subject_id / kind
        folder.mkdir(parents=True, exist_ok=True)
        
        img_rgb = create_synthetic_portrait(is_fake=is_fake)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        file_path = folder / f"sample_{i:03d}.jpg"
        cv2.imwrite(str(file_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
    print("[Done] Generated sample dataset.")


if __name__ == "__main__":
    generate_sample_dataset()
