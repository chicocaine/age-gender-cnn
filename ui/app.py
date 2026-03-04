"""
FastAPI backend for the Age-Gender CNN demo UI.

Endpoints:
    POST /predict   – accepts an image upload, detects face(s), returns predictions
    GET  /health    – liveness check

Run with:
    uvicorn ui.app:app --reload --port 8000
  or from ui/:
    uvicorn app:app --reload --port 8000
"""

import sys
from pathlib import Path

# Make src/ importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference.predict import FacePredictor

app = FastAPI(title="Age-Gender CNN Demo", version="1.0.0")

# Allow the Next.js dev server (port 3000) and production build
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
_predictor: FacePredictor | None = None


@app.on_event("startup")
async def _load_model():
    global _predictor
    _predictor = FacePredictor()
    print(f"✓ Model loaded: {_predictor.model_path.name}")
    print(f"  Device: {_predictor.device}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _predictor.model_path.name if _predictor else None,
        "device": str(_predictor.device) if _predictor else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image upload and return age/gender predictions for all detected faces."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    result = _predictor.predict_from_bytes(image_bytes)
    return result
