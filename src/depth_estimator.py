"""Estimation de profondeur monoculaire 100 % locale (Depth Anything V2 Small, Apache-2.0).

Le modèle (~100 Mo) est téléchargé une seule fois depuis Hugging Face dans models/hf/,
puis tout tourne hors ligne sur le processeur (~1,5 s par image 576×768).

Sortie : carte de « disparité relative » (plus la valeur est grande, plus c'est PROCHE de
la caméra). Les valeurs n'ont pas d'unité : on ne compare que des zones d'une même image.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

from src import MODELS_DIR

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    cache = MODELS_DIR / "hf"
    try:  # hors ligne si déjà téléchargé
        proc = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        cache.mkdir(parents=True, exist_ok=True)
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    return proc, model.eval()


def depth_available() -> bool:
    try:
        import transformers  # noqa: F401
        _load()
        return True
    except Exception:
        return False


def estimate_depth(img: np.ndarray) -> np.ndarray:
    """Disparité relative (float32, même taille que l'image) : grand = proche."""
    import torch
    from PIL import Image

    proc, model = _load()
    inputs = proc(images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), return_tensors="pt")
    with torch.no_grad():
        d = model(**inputs).predicted_depth
    d = torch.nn.functional.interpolate(d[None], size=img.shape[:2], mode="bicubic", align_corners=False)
    return d[0, 0].numpy().astype(np.float32)
