@echo off
echo ========================================================
echo   Starting Deepfake Detector (Backend + Frontend)
echo ========================================================
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
start "Deepfake Detector Backend" cmd /k "uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
cd frontend
start "Deepfake Detector Frontend" cmd /k "npm run dev"
cd ..
echo Applications launched!
echo Backend:  http://127.0.0.1:8000 (API docs: http://127.0.0.1:8000/docs)
echo Frontend: http://localhost:5173
