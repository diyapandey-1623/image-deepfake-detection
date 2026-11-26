# Reproduce Guide: Deepfake Detector

Follow these steps exactly to run this project on your own laptop and get the same results as the original author.

---

## 0. What "same result" means

Results are considered the same when:

- Every predicted label (real/fake) matches the reference predictions.
- Every probability differs by less than **0.005** (0.5%).

Tiny differences in the last digits are normal between different CPUs/GPUs because of floating-point math. This is not a bug.

---

## 1. Prerequisites

| Item | Requirement |
|---|---|
| OS | Windows 10/11, macOS, or Linux |
| Python | 3.10 – 3.13 (Tested on Python 3.13.7) |
| Node.js | 18, 20, or 24 (Tested on v24.14.0) |
| Git | Any recent version |
| Disk space | About 5 GB free (more if you download the Kaggle dataset) |
| RAM | 8 GB minimum |
| GPU | Not required. CPU is enough to run the website and evaluation. A GPU is only useful for training. |
| Accounts | Hugging Face (free), Kaggle (free, only needed for the dataset) |

Check your versions:

```bash
python --version
node --version
git --version
```

---

## 2. Clone the repository

```bash
git clone https://github.com/diyapandey-1623/image-deepfake-detection.git
cd image-deepfake-detection
```

---

## 3. Create a virtual environment and install packages

**Windows (PowerShell or CMD):**

```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

Frontend packages:

```bash
cd frontend
npm ci
cd ..
```

Use `npm ci` (not `npm install`) so the exact versions in `package-lock.json` are installed.

---

## 4. Create your environment file

**Windows:** `copy .env.example .env`  
**Mac / Linux:** `cp .env.example .env`

Open `.env` and fill in your own values (for example `HF_TOKEN` if the model is gated). Never share or commit this file.

---

## 5. Get the model

The model is downloaded automatically from Hugging Face the first time the backend starts.

- Model repo: `dima806/deepfake_vs_real_image_detection`
- Pinned revision (commit hash): `29e4cf9efc543845610045f6ba7e88e5cf9d9301`

The first start is slow because of this download. It is cached in `models/best_model_vit/` afterwards.

To download it manually:

```bash
python scripts/download_model.py
```

---

## 6. Get the dataset (only needed for evaluation and training)

You do NOT need the dataset to run the website. You need it to reproduce the evaluation numbers.

1. Open the Kaggle dataset: [140k Real and Fake Faces Dataset on Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
2. Create a Kaggle API token: Kaggle, then Settings, then API, then "Create New Token". This downloads `kaggle.json`.
3. Place `kaggle.json` here:
   - Windows: `C:\Users\<your-name>\.kaggle\kaggle.json`
   - Mac / Linux: `~/.kaggle/kaggle.json`
4. Download the dataset into `data/raw/` (follow the command in the Kaggle dataset page or use `kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw --unzip`).

### Build the exact same sample

```bash
python scripts/make_sample.py --seed 42
```

This creates `data/small/` with:

| Split | Real | Fake | Total |
|---|---|---|---|
| train | 350 | 350 | 700 |
| valid | 75 | 75 | 150 |
| test | 250 | 250 | 500 |

> Note: this project uses a balanced 500-image test set (250 real, 250 fake) and 1,350 total curated samples matching `data/small/manifest.csv`.

### Verify your sample matches the original

```bash
python scripts/verify_data.py
```

This compares every file against `data/small/manifest.csv` (SHA256 hashes). It must report no mismatches.

---

## 7. Run the evaluation

```bash
python src/evaluate.py
```

This prints accuracy, precision, recall, F1 and a confusion matrix for `data/small/test`.

---

## 8. Run the automatic reproducibility check

```bash
python scripts/reproduce_check.py
```

This script:

1. Prints your environment (OS, Python, torch, transformers, device, model revision).
2. Runs the model on the test set (and demo images).
3. Compares your predictions with `reference_results/expected_predictions.csv` and `demo_images/expected_predictions.csv`.
4. Prints **PASS** or **FAIL** with a list of any mismatches.

**PASS means your results match the original.**

---

## 9. Start the website

Open two terminals (activate the venv in the first one).

**Terminal 1: backend**

```bash
uvicorn backend.main:app --reload
```

**Terminal 2: frontend**

```bash
cd frontend
npm run dev
```

Open the URL printed by the frontend (usually `http://localhost:5173`).

Quick test:

1. Go to the Detect page.
2. Upload an image from `demo_images/` (10 real and 10 fake samples are included).
3. Compare the result with `demo_images/expected_predictions.csv` or `reference_results/expected_predictions.csv`.

---

## 9b. Or run everything with scripts / Docker

```bash
setup.bat   (Windows)     or     bash setup.sh   (Mac / Linux)
run.bat     (Windows)     or     bash run.sh     (Mac / Linux)
```

Docker (no Python/Node install needed):

```bash
docker compose up
```

---

## 10. Expected results (fill these from the real run)

These numbers come directly from actually running `src/evaluate.py` on the original author's machine using seed `42`.

| Metric | Expected value |
|---|---|
| Accuracy | **55.40%** (0.5540) |
| Precision (fake) | **55.51%** (0.5551) |
| Recall (real) | **56.40%** (0.5640) |
| Recall (fake) | **54.40%** (0.5440) |
| F1 | **54.95%** (0.5495) |
| Confusion matrix | `[[141, 109], [114, 136]]` |

*Confusion Matrix Details:*
- True Real: **141** predicted Real, **109** predicted Fake
- True Fake: **114** predicted Real, **136** predicted Fake

*Demo Images Set (`demo_images/` — 20 samples committed in repo):*
- Accuracy: **55.00%** (11/20)
- Precision (fake): **57.14%**
- Recall (real): **70.00%** (7/10)
- Recall (fake): **40.00%** (4/10)
- F1: **47.06%**
- Confusion matrix: `[[7, 3], [6, 4]]`

Tolerance: same predicted labels (100% agreement), probabilities within 0.005.

---

## 11. Settings that keep results identical

The project fixes these so every laptop behaves the same:

- Random seed: **42** (set in `src/seed.py`)
- Deterministic algorithms enabled in PyTorch (`torch.use_deterministic_algorithms(True)`)
- Pinned package versions in `backend/requirements.txt` and `frontend/package-lock.json`
- One shared config for class mapping, preprocessing and the decision threshold (`0.50`)
- A fixed list of test images in `data/small/manifest.csv` and `reference_results/manifest.csv`
- A pinned model revision on Hugging Face (`29e4cf9efc543845610045f6ba7e88e5cf9d9301`)

Do not change these if you want identical results.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Install Python, tick "Add to PATH", reopen the terminal. |
| `pip install` fails on torch | Install the CPU build: `pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu` |
| Port already in use | Close the other app, or start with another port, for example `uvicorn backend.main:app --port 8001`. |
| First start is very slow | The model is downloading from Hugging Face. Wait for it to finish. |
| `401` or "gated model" error | Create a Hugging Face token, put it in `.env` as `HF_TOKEN`, and accept the model terms on its page. |
| `verify_data.py` shows mismatches | Re-run `make_sample.py` with `--seed 42` on a freshly downloaded dataset. |
| `reproduce_check.py` shows FAIL | Compare the environment report with the author's. The most common cause is different package versions. |
| Predictions differ slightly in the last digits | Normal between CPU and GPU. Only differences above 0.005 matter. |

---

## 13. Limitations (read this)

- This detector gives a probability, not proof. Do not use it as the only basis for accusing anyone.
- It was trained and tested on one dataset of face images. Accuracy on photos from other sources (phones, social media, compressed images) can be much lower.
- Only analyze images you have the right to use.
