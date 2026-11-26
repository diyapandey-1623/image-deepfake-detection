# FakeLens: Deepfake Detection & Facial Forensics System

An end-to-end, multimodal deepfake detection application featuring Vision Transformer (ViT) backbones, MTCNN facial landmark alignment, spatial & frequency forensic analysis, Grad-CAM explainability with anatomical regional influence analysis (Red, Yellow, Blue), an automated SQLite audit log, downloadable PDF forensic reports, and a modern editorial React + Tailwind web interface.

---

## 1. Project Overview & Features

- **Model-Agnostic Deep Learning:** Built on Google's Vision Transformer (`google/vit-base-patch16-224` / `dima806/deepfake_vs_real_image_detection`) with runtime fallback to custom PyTorch architectures (`EfficientNet-B0`, `ResNet-18`).
- **Facial Landmark Cropping:** Automated face localization and standardized $+20\%$ boundary margin cropping using MTCNN (`facenet-pytorch`).
- **Explainable AI (Grad-CAM):** Native ViT activation mapping with spatial tensor reshaping (`14x14` token grid), projecting attention heatmaps directly over facial geometry.
- **Regional Anatomical Breakdown:** Evaluates activation across 5 core facial regions:
  - 🟥 **High Influence (Red, $\ge 50\%$):** Focal generative boundary anomalies or primary manipulation features.
  - 🟨 **Moderate Influence (Yellow, $25\% - 50\%$):** Secondary facial contextual structures.
  - 🟦 **Low Influence (Blue, $< 25\%$):** Normal organic skin baseline & background.
- **Forensic Integrity Verifications:**
  - **Error Level Analysis (ELA):** Detects localized JPEG re-quantization inconsistencies.
  - **Metadata / EXIF Audit:** Scans for software manipulation tags (Photoshop, Stable Diffusion, DALL-E, etc.).
- **Forensic PDF Export:** Professional evidence-backed PDF report downloadable for any scan via ReportLab.

---

## 2. Directory Structure

```text
deepfake-detector/
├── data/                     # Dataset storage (gitignored, kept via .gitkeep)
│   ├── raw/                  # Full 140,000 Kaggle dataset
│   ├── small/                # Balanced evaluation set (train: 700, valid: 150, test: 150)
│   └── sample_1000/          # 1,000-image balanced test set (500 Real, 500 Fake)
├── models/                   # Model weight checkpoints (gitignored, kept via .gitkeep)
│   ├── best_model_vit/       # Fine-tuned Vision Transformer checkpoint
│   ├── pretrained_vit/       # Base pretrained ViT weights
│   └── model.pth             # PyTorch CNN weights (EfficientNet-B0)
├── src/                      # ML pipeline source code
│   ├── train.py              # Fine-tuning script for Vision Transformers
│   ├── evaluate.py           # Benchmark script with confusion matrix & error logs
│   ├── predict.py            # CLI prediction tool
│   ├── preprocessing/        # MTCNN face alignment & margin cropping
│   ├── features/             # Spatial, frequency (FFT), and ELA forensic extractors
│   └── models/               # PyTorch model definitions & wrappers
├── backend/                  # FastAPI REST API
│   ├── main.py               # API application & lifespan model caching
│   ├── config.py             # Centralized configuration & environment loader
│   ├── database.py           # SQLite database persistence (scan_history)
│   ├── pdf_report.py         # Forensic PDF report generator (ReportLab)
│   ├── schemas.py            # Pydantic response models
│   └── model/
│       ├── inference.py      # Unified model loader, ViT Grad-CAM & region analyzer
│       └── checks.py         # MTCNN, ELA, and metadata forensic modules
├── frontend/                 # React + Vite + Tailwind CSS Web Application
│   ├── src/
│   │   ├── components/       # UI components (ImageCompare, ReasonsList, DemoUpload, etc.)
│   │   ├── pages/            # App pages (Overview, Home/Detect, Result, History)
│   │   └── lib/              # API client & verdict formatting utilities
│   ├── package.json          # Node dependencies
│   └── vite.config.js        # Vite build & backend proxy configuration
├── configs/                  # Hyperparameter configuration files
├── outputs/                  # Evaluation metrics, prediction CSVs, and plots
├── tests/                    # Automated integration & unit tests
├── .env.example              # Environment variables template
├── .gitignore                # Production ignore rules (datasets, weights, node_modules)
└── requirements.txt          # Python dependencies
```

---

## 3. Dataset Download (Kaggle)

This project is trained and evaluated on the **140k Real and Fake Faces** dataset:

- **Kaggle Dataset Link:** [https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

### How to set up the data:
1. Download the dataset zip from Kaggle using the link above or the Kaggle CLI:
   ```bash
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces
   ```
2. Extract the archive into the `data/` directory so the folder structure is:
   ```text
   data/
   ├── real_vs_fake/
   │   ├── train/
   │   │   ├── real/
   │   │   └── fake/
   │   ├── valid/
   │   │   ├── real/
   │   │   └── fake/
   │   └── test/
   │       ├── real/
   │       └── fake/
   ```

---

## 4. Models: Where to Get or Train

### Option A: Use Pretrained Vision Transformer (Recommended)
By default, the backend automatically downloads and caches the pretrained Vision Transformer model from the Hugging Face Hub:
- **Hugging Face Model:** [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
- **Architecture:** `google/vit-base-patch16-224` fine-tuned for Real vs. Fake face binary classification.

### Option B: Fine-Tune Locally
To fine-tune your own ViT checkpoint on your local dataset:
```bash
python src/train.py --data_dir data/small --output_dir models/best_model_vit --epochs 5 --batch_size 16 --lr 2e-5
```

---

## 5. Setup & Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Node.js 18+ and npm

### 1. Install Backend Dependencies
```bash
# Clone the repository
git clone https://github.com/diyapandey-1623/image-deepfake-detection.git
cd image-deepfake-detection

# Install Python requirements
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

---

## 6. How to Run the Application

### 1. Start the FastAPI Backend Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- **Backend API:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Start the React Frontend Development Server
```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
- **Web Interface:** [http://localhost:5173](http://localhost:5173)

---

## 7. Model Evaluation & CLI Tools

### Single-Image CLI Prediction
```bash
python src/predict.py path/to/face_photo.jpg
```

### Evaluate on Test Dataset
```bash
python src/evaluate.py --test_dir data/small/test --output_dir outputs
```

### Evaluate on 1,000 Balanced Samples
```bash
python src/test_sample_1000.py
```

---

## 8. Limitations & Accuracy Notes

1. **Accuracy Range:** The Vision Transformer achieves **~88% – 93% accuracy** on standard in-distribution benchmark portrait datasets (StyleGAN2, StyleGAN3, ProGAN).
2. **Social Media Compression:** Heavy lossy compression (such as re-saved WhatsApp or Twitter JPEGs at $Q < 50$) degrades subtle high-frequency spatial clues and can lower classifier confidence into the borderline region.
3. **Out-of-Distribution Synthesis:** Novel generative pipelines (e.g. Midjourney v6, Flux.1, or heavily post-processed composite renders) may exhibit different generative signatures from the training set.
4. **Forensic Best Practice:** Predictions are calibrated probabilities ($P_{\text{fake}} \in [0.0, 1.0]$) and should be corroborated with Error Level Analysis (ELA) and cryptographic metadata before journalistic or legal attribution.
