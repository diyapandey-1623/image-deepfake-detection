#!/usr/bin/env bash
echo "========================================================"
echo "  Starting Deepfake Detector (Backend + Frontend)"
echo "========================================================"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..
echo "Applications launched!"
echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://localhost:5173"
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
