"""
Inference pipeline for age-gender prediction.

Handles face detection (OpenCV Haar cascade), face cropping,
preprocessing, and model inference in a single class.
"""

import base64
import io
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Allow importing from src/ when called directly
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.preprocessing import get_inference_transforms
from models.multitask_model import AgeGenderModel


def _find_best_model(models_dir: Path) -> Path:
    """Return the .pth file with the best (lowest) age_mae from experiment results."""
    import json
    import re

    exp_dir = models_dir.parent / "experiments"
    id_pattern = re.compile(r"^(exp\d+_[^_]+(?:_[^_]+)*)_results")
    latest: dict[str, Path] = {}
    for fpath in sorted(exp_dir.glob("exp*_results*.json")):
        m = id_pattern.match(fpath.stem)
        if m:
            latest[m.group(1)] = fpath

    best_id, best_mae = None, float("inf")
    for exp_id, fpath in latest.items():
        try:
            data = json.loads(fpath.read_text())
            mae = data["metrics"]["age_mae"]
            if mae < best_mae:
                best_mae, best_id = mae, exp_id
        except Exception:
            pass

    if best_id is None:
        raise FileNotFoundError("No valid experiment result files found.")
    return models_dir / f"{best_id}_best.pth"


class FacePredictor:
    """End-to-end pipeline: detect face → crop → predict age & gender.

    Args:
        model_path: Path to a .pth checkpoint. If None, the best checkpoint
                    is selected automatically from the experiments/ directory.
        device:     Torch device. Defaults to CUDA if available.
    """

    # Haar cascade bundled with opencv-python
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Resolve model path
        if model_path is None:
            models_dir = Path(__file__).resolve().parent.parent.parent / "models"
            model_path = _find_best_model(models_dir)

        # Load model
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model = AgeGenderModel().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model_path = model_path

        # Preprocessing transform (same as evaluation)
        self.transform = get_inference_transforms()

        # Face detector
        self.detector = cv2.CascadeClassifier(self._CASCADE_PATH)
        if self.detector.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from {self._CASCADE_PATH}. "
                "Ensure opencv-python is installed correctly."
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_from_pil(self, pil_image: Image.Image) -> dict:
        """Run the full pipeline on a PIL image.

        Returns a dict with:
            success       – bool
            faces         – list of per-face results (see below)
            error         – str if success is False
        
        Each face result:
            age           – predicted age (float)
            gender        – 'Male' or 'Female'
            gender_prob   – sigmoid probability for Female class (float 0-1)
            bbox          – [x, y, w, h] in original image pixels
            face_b64      – base64-encoded JPEG of the cropped face (224×224)
        """
        img_rgb = np.array(pil_image.convert("RGB"))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        faces_xywh = self.detector.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )

        if len(faces_xywh) == 0:
            return {"success": False, "faces": [], "error": "No face detected."}

        results = []
        for x, y, w, h in faces_xywh:
            # Crop with a small padding
            pad = int(min(w, h) * 0.10)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_rgb.shape[1], x + w + pad)
            y2 = min(img_rgb.shape[0], y + h + pad)
            face_crop = Image.fromarray(img_rgb[y1:y2, x1:x2])

            age, gender_prob = self._run_model(face_crop)

            results.append(
                {
                    "age": round(float(age), 1),
                    "gender": "Female" if gender_prob >= 0.5 else "Male",
                    "gender_prob": round(float(gender_prob), 4),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "face_b64": self._pil_to_b64(face_crop.resize((224, 224))),
                }
            )

        return {"success": True, "faces": results, "error": None}

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """Convenience wrapper that accepts raw image bytes."""
        pil = Image.open(io.BytesIO(image_bytes))
        return self.predict_from_pil(pil)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _run_model(self, face_pil: Image.Image) -> tuple[float, float]:
        """Preprocess a cropped face PIL image and run the model."""
        face_np = np.array(face_pil.convert("RGB"))
        augmented = self.transform(image=face_np)
        tensor = augmented["image"].unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            age_pred, gender_pred = self.model(tensor)

        return age_pred.item(), gender_pred.item()

    @staticmethod
    def _pil_to_b64(pil_image: Image.Image) -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
