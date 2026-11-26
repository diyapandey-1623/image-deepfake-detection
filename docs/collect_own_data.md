# Dataset Collection and Ethics Guide

## 1. Ethical Standards and Explicit Consent
Collecting data for biometric and face forensic research requires strict adherence to ethical and legal privacy guidelines:
1. **Explicit Informed Consent**: Obtain written or recorded verbal consent from every individual photographed. The consent must explicitly state that the images will be utilized strictly for deepfake detection research and model evaluation.
2. **Right to Revocation**: Ensure individuals have a documented channel to request deletion of their data at any stage.
3. **Privacy & Anonymization**: Do not use real names in folder structures or filenames. Use pseudo-anonymized identifiers (e.g., `subject_01`, `subject_02`).

---

## 2. Capturing Real Imagery (Pristine Reference Data)
To ensure the detector does not overfit to specific camera sensors or lighting setups:
- **Diversity of Subjects**: Capture diverse genders, skin tones, age groups, and facial features (with and without glasses, varied facial hair).
- **Lighting Conditions**:
  - Natural outdoor sunlight (diffuse overcast, direct sun)
  - Indoor warm incandescent lighting
  - Indoor cool fluorescent/LED lighting
  - Low-light / evening indoor environments
- **Capture Devices**: Use multiple physical devices (e.g., smartphones, webcams, DSLR) to prevent the frequency branch from overfitting to a single camera sensor's noise profile.
- **Pose & Expression**: Capture forward-facing neutral portraits, subtle head rotations (+/- 20 degrees), smiling, talking, and varied facial expressions.
- **Resolution**: High-resolution uncompressed formats (PNG or high-quality JPEG >= 95%).

---

## 3. Creating or Obtaining Synthetic/Deepfake Imagery
For model training and evaluation:
- **Generative Diffusion Models**: Generate synthetic portraits using state-of-the-art diffusion models (e.g. Stable Diffusion, Flux, Midjourney).
- **GAN Synthetics**: Utilize GAN-generated faces (e.g. StyleGAN2 / StyleGAN3).
- **Face Swapping**: Apply open facial swap tools (e.g. Roop, SimSwap, InsightFace) swapping identity between collected subjects.
- **Blending / Inpainting**: Face inpainting and cosmetic retouching.

---

## 4. Directory Organization
Organize raw data into the `data/raw/` folder as follows:

```
data/raw/
  subject_001/
    real/
      img_001.jpg
      img_002.jpg
    fake_diffusion/
      img_fake_001.jpg
    fake_faceswap/
      img_swap_001.jpg
  subject_002/
    real/
      img_001.jpg
    fake_stylegan/
      img_fake_001.jpg
```

---

## 5. Generating labels.csv
Run the automated indexing script:
```bash
python build_labels.py --data_dir data/raw --output_csv data/labels.csv
```

The resulting `labels.csv` ensures subject-based splitting (70% train / 15% validation / 15% test) without identity leakage between sets.
