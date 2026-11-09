"""Génère un jeu de démonstration synthétique (rayon de pharmacie).

    python scripts/generate_demo_data.py

Crée :
    data/references/rayon_01_plein.jpg       rayon complet (référence)
    data/captures/rayon_01_actuel.jpg        photo du jour (cadrage légèrement différent)
    data/captures/visite_rayon_01.mp4        vidéo en travelling avec flou de bougé
    data/catalogue/<produit>/*.jpg           vignettes pour le catalogue visuel (fallback)
    data/demo_verite_terrain.json            réponses attendues (pour les tests)
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import CAPTURES_DIR, CATALOGUE_DIR, DATA_DIR, REFERENCES_DIR  # noqa: E402

W, H = 1600, 1000
MARGIN_X = 80
SHELF_TOPS = [40, 360, 680]          # haut de chaque niveau
SHELF_BOTTOMS = [320, 640, 960]      # planche (les boîtes reposent dessus)
BOARD = 22

# (nom, couleur BGR, largeur, hauteur, colonnes, empilement)
SHELVES = [
    [("DOLIPRANE 1000mg", (40, 200, 250), 92, 125, 4, 2),
     ("EFFERALGAN 500mg", (200, 120, 30), 100, 130, 3, 1),
     ("SPASFON LYOC 80mg", (90, 40, 180), 85, 120, 3, 1)],
    [("DAFALGAN 1g", (60, 60, 200), 95, 135, 3, 1),
     ("GAVISCON 24 sach.", (70, 170, 70), 100, 120, 2, 2),
     ("SMECTA 30 sach.", (190, 160, 230), 105, 140, 3, 1),
     ("IBUPROFENE 400mg", (150, 150, 40), 90, 125, 3, 1)],
    [("VITAMINE C 1g", (0, 120, 240), 88, 130, 4, 1),
     ("STREPSILS miel", (30, 90, 30), 95, 125, 3, 1),
     ("HUMEX rhume", (160, 60, 120), 100, 135, 3, 1)],
]

# Boîtes retirées pour l'état « actuel » : {(nom): indices des boîtes retirées}
REMOVED = {
    "DOLIPRANE 1000mg": [4, 5, 6],      # 3 boîtes du rang du haut → 5/8
    "SPASFON LYOC 80mg": [0, 1, 2],     # rupture totale → 0/3
    "SMECTA 30 sach.": [2],             # 2/3
}


def draw_box(canvas, x, y, w, h, name, color, seed):
    rng = np.random.default_rng(seed)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
    darker = tuple(int(c * 0.55) for c in color)
    # motif propre au produit (identique sur toutes ses boîtes)
    for _ in range(14):
        cx, cy = rng.integers(6, w - 6), rng.integers(int(h * 0.62), h - 6)
        cv2.circle(canvas, (x + int(cx), y + int(cy)), int(rng.integers(2, 6)), darker, -1)
    cv2.rectangle(canvas, (x + 4, y + 6), (x + w - 4, y + int(h * 0.12)), darker, -1)
    band_y1, band_y2 = y + int(h * 0.28), y + int(h * 0.56)
    cv2.rectangle(canvas, (x + 5, band_y1), (x + w - 5, band_y2), (250, 250, 250), -1)
    words = name.split(" ")
    lines = [words[0], " ".join(words[1:])] if len(words) > 1 else [name]
    for k, line in enumerate(lines):
        fs = 0.36
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = x + max(6, (w - tw) // 2)
        ty = band_y1 + 14 + k * 15
        cv2.putText(canvas, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (25, 25, 25), 2)


def render(removed: dict[str, list[int]] | None = None, noise_seed: int = 0):
    rng = np.random.default_rng(noise_seed)
    img = np.full((H, W, 3), (205, 214, 222), np.uint8)
    truth = []
    for s_idx, (top, bottom) in enumerate(zip(SHELF_TOPS, SHELF_BOTTOMS)):
        # fond de la case (plus sombre = profondeur)
        cv2.rectangle(img, (MARGIN_X - 20, top), (W - MARGIN_X + 20, bottom), (150, 158, 165), -1)
        # planche
        cv2.rectangle(img, (MARGIN_X - 40, bottom), (W - MARGIN_X + 40, bottom + BOARD), (40, 70, 110), -1)
        # étiquettes de prix sur la planche
        x = MARGIN_X
        col = 0
        for p_idx, (name, color, bw, bh, cols, stack) in enumerate(SHELVES[s_idx]):
            col += 1
            k = 0
            seed = zlib.crc32(name.encode())
            cv2.rectangle(img, (x, bottom + 4), (x + 70, bottom + BOARD - 4), (245, 245, 245), -1)
            cv2.putText(img, name.split(" ")[0][:9], (x + 3, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 0), 1, cv2.LINE_AA)
            for level in range(stack):
                for c in range(cols):
                    bx = x + c * (bw + 8)
                    by = bottom - (level + 1) * bh - level * 6
                    if not (removed and k in removed.get(name, [])):
                        draw_box(img, bx, by, bw, bh, name, color, seed)
                    k += 1
            truth.append({"produit": name, "etagere": s_idx + 1, "colonne": col,
                          "capacite": cols * stack,
                          "manquants": len((removed or {}).get(name, []))})
            x += cols * (bw + 8) + 26
    noise = rng.normal(0, 3, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, truth


def perturb(img, angle=1.2, scale=0.96, tx=18, ty=-12, gain=1.08, bias=6):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def motion_blur(img, k=21):
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def make_video(scene, path, n_frames=150, fps=25, win_w=1100):
    vw, vh = int(win_w * 0.8) // 2 * 2, int(H * 0.8) // 2 * 2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    sharp = {10, 50, 90, 130, 149}
    for i in range(n_frames):
        x0 = int(round(i / (n_frames - 1) * (W - win_w)))
        frame = cv2.resize(scene[:, x0:x0 + win_w], (vw, vh), interpolation=cv2.INTER_AREA)
        if i not in sharp:
            frame = motion_blur(frame, 15 + (i % 3) * 6)
        writer.write(frame)
    writer.release()


def main():
    ref, truth = render(None, noise_seed=1)
    cur_scene, truth_cur = render(REMOVED, noise_seed=2)
    ref_path = REFERENCES_DIR / "rayon_01_plein.jpg"
    cur_path = CAPTURES_DIR / "rayon_01_actuel.jpg"
    vid_path = CAPTURES_DIR / "visite_rayon_01.mp4"
    cv2.imwrite(str(ref_path), ref, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(cur_path), perturb(cur_scene), [cv2.IMWRITE_JPEG_QUALITY, 92])
    make_video(perturb(cur_scene, angle=0.5, scale=1.0, tx=0, ty=0, gain=1.03, bias=3), vid_path)

    # Catalogue visuel : une vignette par produit, extraite du rendu de référence
    for name, color, bw, bh, *_ in [p for s in SHELVES for p in s]:
        d = CATALOGUE_DIR / name.replace("/", "-")
        d.mkdir(parents=True, exist_ok=True)
        tile = np.full((bh + 10, bw + 10, 3), 255, np.uint8)
        draw_box(tile, 5, 5, bw, bh, name, color, zlib.crc32(name.encode()))
        cv2.imwrite(str(d / "face.jpg"), tile)

    expected = [t for t in truth_cur if t["manquants"]]
    (DATA_DIR / "demo_verite_terrain.json").write_text(
        json.dumps({"blocs": truth_cur, "anomalies_attendues": expected}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"Référence : {ref_path}\nPhoto     : {cur_path}\nVidéo     : {vid_path}")
    print("Anomalies attendues :")
    for e in expected:
        print(f"  Ét.{e['etagere']} Col.{e['colonne']} {e['produit']}: "
              f"{e['capacite'] - e['manquants']}/{e['capacite']}")


if __name__ == "__main__":
    main()
