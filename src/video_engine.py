"""Module 1 — Extraction des frames nettes d'une vidéo (filtre anti-flou laplacien).

Pour chaque fenêtre temporelle de `interval_sec`, seule la frame dont la variance du
Laplacien est la plus élevée (= la plus nette) est conservée sur disque.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from src import CAPTURES_DIR

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def sharpness_score(image_bgr: np.ndarray) -> float:
    """Variance du Laplacien : élevée = image nette, faible = image floue."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sharp_frames(
    video_path: str,
    interval_sec: float = 1.5,
    output_dir: str | Path | None = None,
    min_sharpness: float = 0.0,
    max_width: int = 1920,
) -> list[str]:
    """Extrait la frame la plus nette de chaque fenêtre de `interval_sec` secondes.

    Args:
        video_path: chemin de la vidéo.
        interval_sec: taille de la fenêtre temporelle.
        output_dir: dossier de sortie (défaut : data/captures/frames_<nom_video>/).
        min_sharpness: frames dont le score est inférieur sont ignorées même si elles
            sont les meilleures de leur fenêtre (fenêtre entièrement floue).
        max_width: les frames plus larges sont redimensionnées (accélère la suite).

    Returns:
        Liste des chemins des frames sauvegardées, dans l'ordre chronologique.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    window = max(1, int(round(fps * interval_sec)))

    # Nom de dossier sûr (Windows refuse espaces/points finaux, accents ok)
    stem = re.sub(r"[^\w-]+", "_", Path(video_path).stem, flags=re.UNICODE).strip("_") or "video"
    out_dir = Path(output_dir) if output_dir else CAPTURES_DIR / f"frames_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    saved: list[str] = []
    best_score, best_frame, best_idx = -1.0, None, -1

    def flush() -> None:
        nonlocal best_score, best_frame, best_idx
        if best_frame is not None and best_score >= min_sharpness:
            path = out_dir / f"{stem}_f{best_idx:06d}.jpg"
            cv2.imwrite(str(path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(str(path))
        best_score, best_frame, best_idx = -1.0, None, -1

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        score = sharpness_score(frame)
        if score > best_score:
            best_score, best_frame, best_idx = score, frame.copy(), idx
        idx += 1
        if idx % window == 0:
            flush()
    flush()
    cap.release()
    return saved


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extrait les frames nettes d'une vidéo.")
    p.add_argument("video")
    p.add_argument("--interval", type=float, default=1.5)
    a = p.parse_args()
    for f in extract_sharp_frames(a.video, a.interval):
        print(f)
