@echo off
echo ========================================================
echo   Setting up Deepfake Detector (Windows)
echo ========================================================
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
if not exist .env (
    copy .env.example .env
    echo Created .env from .env.example
)
python scripts/download_model.py
cd frontend
call npm ci
cd ..
echo ========================================================
echo   Setup completed successfully! Run 'run.bat' to start.
echo ========================================================
pause
