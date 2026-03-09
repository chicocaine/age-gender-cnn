@echo off
REM Start FastAPI server for Age-Gender CNN Demo
echo Starting FastAPI server...
cd /d "%~dp0"
call venv312\Scripts\activate.bat
python -m uvicorn ui.app:app --reload --port 8000
