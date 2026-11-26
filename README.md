# FakeLens: Deepfake Detection & Facial Forensics System

An end-to-end deepfake detection application featuring a fine-tuned Vision Transformer (`vishvjeet1623/image-deepfake-detection`), MTCNN facial localization, EXIF orientation normalization, Grad-CAM explainability with 5-region anatomical influence analysis (Red, Yellow, Blue), Error Level Analysis (ELA), metadata inspection, an automated SQLite audit log, downloadable PDF forensic reports, and a React + Tailwind CSS web interface.

---

## 1. Project Overview & Features

- **Fine-Tuned Vision Transformer:** Defaults to [`vishvjeet1623/image-deepfake-detection`](https://huggingface.co/vishvjeet1623/image-deepfake-detection) on Hugging Face Hub (automatically downloaded and cached on first run), with optional local checkpoint loading (`models/best_model_vit/`) or custom PyTorch CNN fallback (`efficientnet_b0`, `resnet18`).
- **Facial Localization & Preprocessing:** Applies EXIF orientation correction (`ImageOps.exif_transpose`), normalizes oversized uploads (`> 1024px`), and crops detected faces with a `+20%` boundary margin using MTCNN (`facenet-pytorch`).
- **Explainable AI (Grad-CAM):** Targets the manipulation class on ViT (`14x14` token grid reshape) or CNN layers to project attention heatmaps over facial geometry.
- **Regional Anatomical Breakdown:** Evaluates mean activation across 5 facial regions:
  - **High Influence (Red, $\ge 50\%$):** Focal generative boundary anomalies or primary manipulation features.
  - **Moderate Influence (Yellow, $25\% - 50\%$):** Secondary facial contextual structures.
  - **Low Influence (Blue, $< 25\%$):** Baseline organic skin & background.
- **Forensic Integrity Checks (on untouched original bytes):**
  - **Error Level Analysis (ELA):** Detects localized JPEG re-quantization inconsistencies.
  - **Metadata / EXIF Audit:** Scans image headers and EXIF tags for editing software or AI generator markers.
- **Audit Log & PDF Export:** Persists scan history in SQLite (`data/scans.db`) and generates downloadable PDF forensic reports via ReportLab.

---

## 2. Directory Structure

```text
image-deepfake-detection/
├── backend/                  # FastAPI REST API
│   ├── main.py               # API endpoints, lifespan caching, EXIF/MTCNN pipeline
│   ├── config.py             # Centralized configuration & environment overrides
│   ├── database.py           # SQLite persistence (data/scans.db)
│   ├── pdf_report.py         # Forensic PDF report generator (ReportLab)
│   ├── schemas.py            # Pydantic response models
│   └── model/
│       ├── inference.py      # HuggingFaceModel & CustomPyTorchModel + Grad-CAM
│       ├── checks.py         # MTCNN face cropping, ELA, and EXIF metadata audit
│       └── samples.py        # Synthetic sample portrait generator
├── frontend/                 # React + Vite + Tailwind CSS Web Application
│   ├── src/
│   │   ├── components/       # UI components (DemoUpload, ImageCompare, VerdictCard, etc.)
│   │   ├── pages/            # App pages (Overview, Home/Detect, Result, History)
│   │   └── lib/              # API client, verdict formatting, and class config
│   ├── package.json
│   └── vite.config.js
├── src/                      # ML training, evaluation & CLI scripts
│   ├── class_config.py       # Shared class mappings, binary threshold, rationale builder
│   ├── vit_transforms.py     # Shared 224x224 ViT preprocessing
│   ├── vit_utils.py          # Shared dataset, model resolution, and confusion matrix utils
│   ├── train.py              # ViT fine-tuning script
│   ├── train_balanced.py     # Balanced ViT fine-tuning with JPEG augmentation
│   ├── evaluate.py           # Batch test-set evaluation script
│   └── predict.py            # Single-image CLI prediction tool
├── tests/                    # Unit, integration, and web benchmark tests
├── start.bat                 # Windows one-click launcher (opens backend & frontend log windows)
├── .env.example              # Environment variables template
└── requirements.txt          # Python dependencies
```

---

## 3. Setup & Installation

### Prerequisites
- **Python 3.10 – 3.13** (tested on **Python 3.11**; PyTorch/MTCNN wheels do not yet support Python 3.14)
- **Node.js 18+** and **npm**

### 1. Clone the Repository
```bash
git clone https://github.com/diyapandey-1623/image-deepfake-detection.git
cd image-deepfake-detection
```

### 2. Create Virtual Environment & Install Python Dependencies
**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
# Windows (PowerShell)
Copy-Item .env.example .env
```
```bash
# macOS / Linux
cp .env.example .env
```
By default, `.env` sets `MODEL_BACKEND=huggingface` and `MODEL_ID=vishvjeet1623/image-deepfake-detection`, which automatically downloads and caches the fine-tuned model on first startup.

---

## 4. How to Run the Application

### Option A: One-Click Launch on Windows (`start.bat`)
Double-click `start.bat` (or run `.\start.bat` from PowerShell). It opens two separate terminal windows so you can monitor live backend and frontend logs:
- **Web Interface:** [http://127.0.0.1:5173](http://127.0.0.1:5173)
- **Backend API & Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Option B: Run Manually in Two Terminals

**Terminal 1 — FastAPI Backend:**
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- **Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

**Terminal 2 — React Frontend:**
```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
- **Web Interface:** [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## 5. Testing, Benchmarking & CLI Tools

### Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Run Web Benchmark (10 Real & Deepfake Web Portraits)
```powershell
.\.venv\Scripts\python.exe tests\benchmark_web_images.py
```

### Single-Image CLI Prediction
```powershell
.\.venv\Scripts\python.exe src\predict.py path\to\face_photo.jpg
```

### Fine-Tune on Your Own Dataset (Optional)
Download the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset into `data/real_vs_fake/` (with `train/`, `valid/`, `test/` containing `real/` and `fake/`), then run:
```powershell
.\.venv\Scripts\python.exe src\train.py --data_dir data\real_vs_fake --output_dir models\best_model_vit --epochs 5 --batch_size 16 --lr 2e-5
```

---

## 6. Model Performance & Limitations

- **Benchmark Accuracy:** The fine-tuned [`vishvjeet1623/image-deepfake-detection`](https://huggingface.co/vishvjeet1623/image-deepfake-detection) checkpoint achieves **93.50% accuracy** (`91.0%` Real Recall, `96.0%` Fake Recall, `93.66%` Fake F1) on the 200-image web evaluation set (`tests/web_eval_200/`) and **100% (10/10)** on `tests/web_benchmark/`.
- **Operational Limitations:** Heavy lossy recompression, extreme lighting, or unseen generative architectures can reduce confidence. Predictions are probabilistic (`P_fake ∈ [0.0, 1.0]`, threshold `0.50`) and should be corroborated with ELA and metadata checks before formal attribution.
