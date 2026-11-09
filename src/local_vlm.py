"""Module 4 — Identification du produit, 100 % locale.

Ordre de priorité :
  1. VLM local via Ollama (``qwen2.5vl:7b`` par défaut) — lit nom, dosage, conditionnement.
  2. Catalogue visuel FAISS (approche de Retail-Shelf-Monitoring / SKUDetector) :
     ``data/catalogue/<Nom du produit>/*.jpg`` → embeddings MobileNetV3 (ImageNet,
     poids téléchargés une fois puis mis en cache dans ``models/torch``) → recherche
     du plus proche voisin. Repli sur histogrammes de couleur si torch est indisponible.
  3. Chaîne vide (l'appelant utilise alors un nom générique « Produit Ét.X Col.Y »).
"""

from __future__ import annotations

import base64
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests

from src import CATALOGUE_DIR, MODELS_DIR

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# NB : le tag officiel Ollama est « qwen2.5vl » (sans tiret entre 2.5 et vl).
DEFAULT_VLM_MODEL = os.environ.get("PHARMA_VLM_MODEL", "qwen2.5vl:7b")
VLM_PROMPT = (
    "Lis le nom du médicament, dosage et conditionnement visible sur cette boîte ou "
    "cette étiquette de rayon. Réponds uniquement par le nom standardisé."
)
CATALOGUE_MIN_SCORE = 0.6
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------------------
def ollama_available(model: str = DEFAULT_VLM_MODEL, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if ":" in model:
            return model in names
        return any(n.split(":")[0] == model for n in names)
    except (requests.RequestException, ValueError):
        return False


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _clean_answer(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"^(nom( standardisé)?\s*:\s*)", "", text, flags=re.I)
    return text.strip(" \"'*`.").strip()


def ask_vlm(image_paths: list[str], prompt: str = VLM_PROMPT, model: str = DEFAULT_VLM_MODEL,
            timeout: float = 180.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_b64(p) for p in image_paths],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return _clean_answer(r.json().get("response", ""))


# --------------------------------------------------------------------------------------
# Catalogue visuel (FAISS)
# --------------------------------------------------------------------------------------
class _Embedder:
    """MobileNetV3-small (torchvision) ou, à défaut, histogramme HSV."""

    def __init__(self) -> None:
        self.kind = "hsv-hist"
        self.model = None
        try:
            os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
            import torch
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            weights = MobileNet_V3_Small_Weights.DEFAULT
            m = mobilenet_v3_small(weights=weights)
            m.classifier = torch.nn.Identity()
            m.eval()
            self.model, self.torch = m, torch
            self.kind = "mobilenet_v3_small"
        except Exception as e:  # pas de torch / pas de réseau au 1er lancement
            print(f"[catalogue] MobileNet indisponible ({e}) → histogrammes de couleur.")

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.model is None:
            hsv = cv2.cvtColor(cv2.resize(img_bgr, (96, 128)), cv2.COLOR_BGR2HSV)
            v = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 4], [0, 180, 0, 256, 0, 256]).flatten()
        else:
            rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            t = self.torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32))
            with self.torch.no_grad():
                v = self.model(t).numpy().flatten()
        v = v.astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


@lru_cache(maxsize=1)
def _embedder() -> _Embedder:
    return _Embedder()


class ProductCatalog:
    """Index visuel des boîtes connues : data/catalogue/<Nom produit>/*.jpg."""

    def __init__(self, catalogue_dir: Path = CATALOGUE_DIR) -> None:
        self.dir = Path(catalogue_dir)
        self.names: list[str] = []
        self.index = None
        self.vectors: np.ndarray | None = None

    def _files(self) -> list[Path]:
        return sorted(p for p in self.dir.glob("*/*") if p.suffix.lower() in IMG_EXT)

    def build(self) -> "ProductCatalog":
        files = self._files()
        if not files:
            return self
        emb = _embedder()
        vecs, names = [], []
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                vecs.append(emb(img))
                names.append(f.parent.name)
        if not vecs:
            return self
        self.vectors = np.vstack(vecs).astype(np.float32)
        self.names = names
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
        except ImportError:
            self.index = None  # recherche numpy
        meta = {"embedder": emb.kind, "n": len(names), "names": sorted(set(names))}
        (MODELS_DIR / "catalogue_index.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return self

    def __len__(self) -> int:
        return len(self.names)

    def search(self, img_bgr: np.ndarray) -> tuple[str, float]:
        if not self.names:
            return "", 0.0
        q = _embedder()(img_bgr)[None]
        if self.index is not None:
            scores, idx = self.index.search(q, 1)
            return self.names[int(idx[0, 0])], float(scores[0, 0])
        sims = self.vectors @ q[0]
        k = int(np.argmax(sims))
        return self.names[k], float(sims[k])


_CATALOG: ProductCatalog | None = None


def get_catalog(rebuild: bool = False) -> ProductCatalog:
    global _CATALOG
    if _CATALOG is None or rebuild:
        _CATALOG = ProductCatalog().build()
    return _CATALOG


# --------------------------------------------------------------------------------------
# Point d'entrée de la spec
# --------------------------------------------------------------------------------------
def identify_product_from_crop(crop_path: str, shelf_label_crop_path: str | None = None,
                               use_vlm: bool = True, model: str = DEFAULT_VLM_MODEL,
                               return_source: bool = False):
    """Identifie un produit à partir d'une vignette (et éventuellement de l'étiquette).

    Returns:
        Le nom standardisé, ou "" si non identifié. Avec ``return_source=True`` :
        ``(nom, source)`` où source ∈ {"vlm", "catalogue", "inconnu"}.
    """
    def ret(name: str, src: str):
        return (name, src) if return_source else name

    if use_vlm and ollama_available(model):
        images = [crop_path] + ([shelf_label_crop_path] if shelf_label_crop_path else [])
        try:
            name = ask_vlm(images, model=model)
            if name:
                return ret(name, "vlm")
        except requests.RequestException as e:
            print(f"[vlm] échec de l'appel Ollama : {e}")

    catalog = get_catalog()
    if len(catalog):
        img = cv2.imread(str(crop_path))
        if img is not None:
            name, score = catalog.search(img)
            if score >= CATALOGUE_MIN_SCORE:
                return ret(name, "catalogue")
    return ret("", "inconnu")


if __name__ == "__main__":
    import sys

    print("Ollama disponible :", ollama_available())
    if len(sys.argv) > 1:
        print(identify_product_from_crop(sys.argv[1], return_source=True))
