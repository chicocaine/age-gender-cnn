# Start FastAPI server for Age-Gender CNN Demo
Write-Host "Starting FastAPI server..." -ForegroundColor Green

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Activate virtual environment and start server
& "$projectRoot\venv312\Scripts\python.exe" -m uvicorn ui.app:app --reload --port 8000
