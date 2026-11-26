# Reproducibility Guide: Deepfake Detection System

This guide provides step-by-step instructions for anyone to clone this repository on their laptop, run the deepfake detection model, and get the **exact same results**. Every metric and prediction in this document comes from an actual run using seed `42`.

---

## 1. System Prerequisites

Ensure your system meets the following specifications before beginning:

| Component | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, macOS 12+, or Ubuntu 20.04+ | Windows 11 / Ubuntu 22.04 |
| **Python** | Python `3.10` – `3.13` (Tested on `3.13.7`) | Python `3.11` or `3.12` |
| **Node.js** | Node.js `v18.0.0`+ (Tested on `v24.14.0`) | Node.js `v20.x LTS` |
| **Package Manager** | `npm` (v9+) or `pnpm` | `npm` |
| **RAM** | 8 GB minimum | 16 GB |
| **Disk Space** | 2 GB free (demo & model) / 10 GB (if downloading full Kaggle raw dataset) | 10 GB SSD |
| **GPU / Hardware** | **None required (CPU is 100% sufficient)** | CPU inference takes ~25 ms/image |

---

## 2. Clone the Repository and Setup Virtual Environment

### Step 2.1: Clone Repository
Open your terminal (PowerShell on Windows, or Bash on macOS/Linux):
```bash
git clone https://github.com/diyapandey-1623/image-deepfake-detection.git
cd image-deepfake-detection
```

### Step 2.2: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

**On macOS / Linux (Bash or Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2.3: Install Pinned Dependencies
The repository uses CPU PyTorch and pinned package versions in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. How to Obtain the Model Weights

You do **not** need to manually search for or train model weights. The system is configured to automatically download and cache the Vision Transformer (ViT) model from the Hugging Face Hub:

- **Model ID**: `dima806/deepfake_vs_real_image_detection`
- **Pinned Commit Hash / Revision**: `29e4cf9efc543845610045f6ba7e88e5cf9d9301`
- **Architecture**: `ViTForImageClassification` (`google/vit-base-patch16-224-in21k` backbone)

When you run `python scripts/reproduce_check.py` or launch the backend, the model downloads automatically on the first execution. If you have a local copy in `models/best_model_vit/`, the system will load it directly.

---

## 4. Immediate Verification (No Kaggle Account Needed)

To allow strangers to test reproducibility immediately without downloading large datasets, **20 demo images (10 Real, 10 Fake)** are committed directly to `demo_images/`.

Run the automated verification script:
```bash
python scripts/reproduce_check.py --target demo
```

This will:
1. Print an **Environment Diagnostic Report** (OS, Python, PyTorch, device, model revision).
2. Execute deterministic inference (`seed=42`).
3. Compare every prediction against `demo_images/expected_predictions.csv`.
4. Validate that labels match 100% and probabilities match within the tolerance `|p_actual - p_expected| <= 0.005`.
5. Output a `PASS` confirmation.

---

## 5. Full Dataset Reproduction (From Kaggle)

If you want to reproduce the full 500-sample test set from scratch:

### Step 5.1: Download Kaggle Dataset
1. Visit the dataset page: [140k Real and Fake Faces Dataset on Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).
2. Download your Kaggle API key:
   - Go to your Kaggle Account Settings (`https://www.kaggle.com/settings`).
   - Click **"Create New Token"** to download `kaggle.json`.
   - Place `kaggle.json` into:
     - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - **macOS/Linux**: `~/.kaggle/kaggle.json`
3. Download and unpack the dataset:
   ```bash
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw --unzip
   ```

### Step 5.2: Generate the Deterministic Sample
Run the sampling script with seed `42`. This selects balanced real and fake images and calculates SHA256 checksums:
```bash
python scripts/make_sample.py --seed 42
```
This generates:
- `data/small/train` (350 real, 350 fake)
- `data/small/valid` (75 real, 75 fake)
- `data/small/test` (250 real, 250 fake)
- `data/small/manifest.csv` (SHA256 checksums for every image)

### Step 5.3: Verify Sample Integrity
Ensure all 1,350 files match the reference hashes:
```bash
python scripts/verify_data.py
```
Expected output:
```text
All 1350 files verified successfully. Data integrity: PASS
```

### Step 5.4: Run Full Reproducibility Verification
```bash
python scripts/reproduce_check.py --target all
```

---

## 6. Expected Results & Benchmark Metrics

All metrics below are derived from actual code execution on the reference dataset using seed `42` and a decision threshold of `0.50`:

### Full Test Set (`data/small/test` — 500 Images)

| Metric | Reference Value | Tolerance / Acceptance Criteria |
| :--- | :--- | :--- |
| **Total Test Images** | 500 (250 Real, 250 Fake) | Exact |
| **Accuracy** | **56.80%** (0.568) | +/- 0.5% |
| **F1-Score (Fake)** | **56.28%** (0.5628) | +/- 0.5% |
| **Precision (Fake)** | **56.97%** (0.5697) | +/- 0.5% |
| **Recall (Fake)** | **55.60%** (0.5560) | +/- 0.5% |
| **Real -> Predicted Real** | **145** | Exact |
| **Real -> Predicted Fake** | **105** | Exact |
| **Fake -> Predicted Real** | **111** | Exact |
| **Fake -> Predicted Fake** | **139** | Exact |

### Demo Set (`demo_images/` — 20 Images)

| Metric | Reference Value | Tolerance / Acceptance Criteria |
| :--- | :--- | :--- |
| **Total Images** | 20 (10 Real, 10 Fake) | Exact |
| **Accuracy** | **60.00%** (12 / 20) | Exact (100% label match) |
| **F1-Score (Fake)** | **55.56%** (0.5556) | Exact |
| **Precision (Fake)** | **62.50%** (0.6250) | Exact |
| **Recall (Fake)** | **50.00%** (0.5000) | Exact |
| **Real -> Predicted Real** | **7** | Exact |
| **Real -> Predicted Fake** | **3** | Exact |
| **Fake -> Predicted Real** | **5** | Exact |
| **Fake -> Predicted Fake** | **5** | Exact |

### Tolerance Rules for Cross-Machine Reproducibility
- **Label Prediction Agreement**: Must be **100.0%**. Every image must yield the exact same predicted class (`Real` vs `Fake`).
- **Probability Delta**: `|actual_probability - expected_probability| <= 0.005`.
  > **Why tolerance is allowed:** Differences $\le 0.005$ are normal across distinct CPU/GPU microarchitectures (e.g. Intel AVX2 vs Apple Silicon M-series vs NVIDIA cuDNN) due to hardware-specific floating-point reduction order in matrix multiplication.

---

## 7. Running the Application (Backend & Frontend)

### Step 7.1: Start FastAPI Backend
In your activated virtual environment:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### Step 7.2: Start React Frontend
In a second terminal window:
```bash
cd frontend
npm install
npm run dev
```
- Web Application: `http://localhost:5173`

### Step 7.3: Test an Image in the Web UI
1. Open `http://localhost:5173` in your browser.
2. Drag and drop any image from `demo_images/real/` or `demo_images/fake/` into the upload zone.
3. Click **"Analyze Image"**.
4. Observe the results:
   - Verdict (`Authentic Image` or `Synthetic / Deepfake`)
   - Confidence score
   - Grad-CAM heatmap visualization
   - Forensic checks (Metadata analysis, Error Level Analysis - ELA)
   - PDF forensic report export button

---

## 8. Determinism Architecture

To guarantee identical results across runs, the repository enforces determinism via `src/seed.py`:
- `random.seed(42)`
- `np.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- `torch.use_deterministic_algorithms(True)`
- Image normalization: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- Image resizing: Bilinear interpolation to `(224, 224)`
- Class mapping: Unified in `src/class_config.py` and `backend/class_config.py` (`0: Real`, `1: Fake`)
- Decision threshold: `0.50`

---

## 9. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `kaggle.json not found` | Kaggle API credentials not in user home directory | Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows). |
| `Port 8000 already in use` | Another instance of uvicorn is running | Kill the existing process: Windows: `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force` |
| `Port 5173 already in use` | Vite selected port 5174 | Close previous Vite server or visit the port Vite outputs in the terminal. |
| `ImportError: cannot import name ...` | Package version drift | Run `pip install -r requirements.txt` inside your clean virtual environment. |
| `CUDA out of memory` | GPU VRAM exhausted | Run on CPU: set `DEVICE=cpu` in `.env` or run on CPU (CPU uses negligible memory, ~1 GB RAM). |
| `reproduce_check.py failed` | Tolerance discrepancy | Check `scripts/reproduce_check.py` output table to see which image failed. Confirm that model weights were loaded from commit `29e4cf9efc543845610045f6ba7e88e5cf9d9301`. |
