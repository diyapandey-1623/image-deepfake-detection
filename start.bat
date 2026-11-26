@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [ERROR] Frontend dependencies not found. Run "npm install" inside frontend\ first.
    pause
    exit /b 1
)

echo Starting FakeLens Backend (http://127.0.0.1:8000)...
start "FakeLens Backend (FastAPI)" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo Starting FakeLens Frontend (http://127.0.0.1:5173)...
start "FakeLens Frontend (Vite)" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo Both servers launched in separate windows so you can view live logs:
echo   Frontend: http://127.0.0.1:5173
echo   Backend:  http://127.0.0.1:8000/docs
echo.
pause
