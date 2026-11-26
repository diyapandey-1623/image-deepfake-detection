#!/usr/bin/env bash
set -e
echo "========================================================"
echo "  Setting up Deepfake Detector (Mac / Linux)"
echo "========================================================"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi
python scripts/download_model.py
cd frontend
npm ci
cd ..
echo "========================================================"
echo "  Setup completed successfully! Run 'bash run.sh' to start."
echo "========================================================"
