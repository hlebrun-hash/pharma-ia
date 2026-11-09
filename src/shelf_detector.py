"""Module 2 — Détection des boîtes et des espaces vides (trous) sur une photo de rayon.

Deux détecteurs, 100 % locaux :
  * ``yolo``      : modèle Ultralytics (.pt / .onnx). Idéalement un modèle entraîné sur
                    SKU-110K (détection dense de produits en rayon) placé dans ``models/``.
  * ``classical`` : segmentation OpenCV (contours rectangulaires). Aucun poids requis,
                    fonctionne bien sur des rayons propres et bien éclairés.
  * ``auto``      : YOLO si des poids « rayon » sont trouvés dans ``models/``, sinon classique.

Le résultat est ensuite structuré en rangées (même hauteur), étagères (rangées
empilées sans planche entre elles) et trous (écart horizontal > largeur médiane).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src import MODELS_DIR

_YOLO_CACHE: dict[str, Any] = {}
RETAIL_WEIGHT_HINTS = ("sku", "shelf", "retail", "rayon", "pharma")


# --------------------------------------------------------------------------------------
# Utilitaires géométriques
# --------------------------------------------------------------------------------------
def box_iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def containment(inner, outer) -> float:
    """Fraction de la surface de `inner` contenue dans `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def nms(boxes: list[list[float]], iou_thr: float = 0.5) -> list[list[float]]:
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept: list[list[float]] = []
    for b in boxes:
        if all(box_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


# --------------------------------------------------------------------------------------
# Détecteurs
# --------------------------------------------------------------------------------------
def find_retail_weights() -> str | None:
    """Cherche des poids de détection « rayon » (SKU-110K…) dans models/ ou via env."""
    env = os.environ.get("PHARMA_DETECTOR_WEIGHTS")
    if env and Path(env).exists():
        return env
    for ext in ("*.pt", "*.onnx"):
        for p in sorted(MODELS_DIR.glob(ext)):
            if any(h in p.stem.lower() for h in RETAIL_WEIGHT_HINTS):
                return str(p)
    return None


def _detect_classical(img: np.ndarray) -> list[list[float]]:
    h, w = img.shape[:2]
    img_area = float(h * w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cands: list[list[float]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(bw * bh)
        if area < img_area * 0.0008 or area > img_area * 0.2:
            continue
        ar = bw / float(bh)
        if ar < 0.15 or ar > 8.0:  # large tolérance : fragments (bandeaux) fusionnés plus bas
            continue
        fill = cv2.contourArea(c) / area
        if fill < 0.8:
            continue
        cands.append([float(x), float(y), float(x + bw), float(y + bh), round(fill, 3)])

    # 1) Supprime les « conteneurs » (cases d'étagère contenant plusieurs boîtes
    #    côte à côte). Une boîte contient elle-même des zones (bandeau d'étiquette,
    #    motif) mais elles sont superposées horizontalement, pas côte à côte.
    def is_container(b) -> bool:
        a_b = (b[2] - b[0]) * (b[3] - b[1])
        kids = [o for o in cands if o is not b
                and (o[2] - o[0]) * (o[3] - o[1]) < 0.7 * a_b and containment(o, b) > 0.9]
        for i, k1 in enumerate(kids):
            for k2 in kids[i + 1:]:
                x_overlap = min(k1[2], k2[2]) - max(k1[0], k2[0])
                if x_overlap < 0.2 * min(k1[2] - k1[0], k2[2] - k2[0]):
                    return True
        return False

    cands = [b for b in cands if not is_container(b)]
    # 2) Supprime les doublons imbriqués (bandeau d'étiquette, contour intérieur).
    cands.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept: list[list[float]] = []
    for b in cands:
        if not any(containment(b, k) > 0.85 for k in kept):
            kept.append(b)
    # 3) Fusionne les fragments d'une même boîte : quand le bord de la boîte se confond
    #    avec la planche, seules ses zones internes (bandeau, motif…) sont trouvées ;
    #    elles ont la même étendue horizontale et sont quasi jointives verticalement.
    boxes = []
    for b in _merge_vertical_fragments(kept):
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw * bh >= img_area * 0.002 and 0.2 <= bw / bh <= 4.0:
            boxes.append(b)
    return nms(boxes, 0.5)


def _merge_vertical_fragments(boxes: list[list[float]], max_gap: float = 8.0) -> list[list[float]]:
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        boxes.sort(key=lambda b: (b[1], b[0]))
        for i in range(len(boxes)):
            a = boxes[i]
            for j in range(i + 1, len(boxes)):
                b = boxes[j]
                wa, wb = a[2] - a[0], b[2] - b[0]
                tol = 0.1 * max(wa, wb)
                same_cols = abs(a[0] - b[0]) < tol and abs(a[2] - b[2]) < tol
                v_gap = max(a[1], b[1]) - min(a[3], b[3])
                new_h = max(a[3], b[3]) - min(a[1], b[1])
                if same_cols and v_gap <= max_gap and new_h / max(wa, wb) < 2.5:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]),
                                max(a[3], b[3]), max(a[4], b[4])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _detect_yolo(img: np.ndarray, weights: str, conf: float) -> list[list[float]]:
    from ultralytics import YOLO

    model = _YOLO_CACHE.get(weights)
    if model is None:
        model = YOLO(weights)
        _YOLO_CACHE[weights] = model
    res = model.predict(img, conf=conf, iou=0.5, max_det=1000, agnostic_nms=True,
                        imgsz=1024, verbose=False)[0]
    names = getattr(res, "names", {}) or {}
    out: list[list[float]] = []
    for xyxy, c, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                            res.boxes.cls.cpu().numpy()):
        if names.get(int(cls), "") == "person":  # modèles COCO génériques
            continue
        out.append([*map(float, xyxy), float(c)])
    return out


def detect_boxes(img: np.ndarray, method: str = "auto", weights: str | None = None,
                 conf: float = 0.25) -> tuple[list[list[float]], str]:
    """Retourne ([x1, y1, x2, y2, score], méthode_utilisée)."""
    if method == "auto":
        # Ordre : poids « rayon » (SKU-110K…) > YOLO générique COCO (détecte les
        # flacons, nettement meilleur que l'OpenCV classique sur de vraies vidéos)
        # > détection classique (rayons propres / démo uniquement).
        generic = MODELS_DIR / "yolov8m.pt"
        weights = weights or find_retail_weights() or (str(generic) if generic.exists() else None)
        method = "yolo" if weights else "classical"
    if method == "yolo":
        weights = weights or find_retail_weights() or str(MODELS_DIR / "yolov8m.pt")
        return _detect_yolo(img, weights, conf), f"yolo:{Path(weights).name}"
    return _detect_classical(img), "classical"


# --------------------------------------------------------------------------------------
# Structuration : rangées, étagères, trous
# --------------------------------------------------------------------------------------
def group_rows(boxes: list[list[float]]) -> list[list[int]]:
    """Regroupe les boîtes en rangées horizontales selon leur centre Y."""
    if not boxes:
        return []
    med_h = float(np.median([b[3] - b[1] for b in boxes]))
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        cy = (boxes[i][1] + boxes[i][3]) / 2
        if rows:
            row_cy = np.mean([(boxes[j][1] + boxes[j][3]) / 2 for j in rows[-1]])
            if abs(cy - row_cy) < 0.5 * med_h:
                rows[-1].append(i)
                continue
        rows.append([i])
    for r in rows:
        r.sort(key=lambda i: boxes[i][0])
    return rows


def group_shelves(boxes: list[list[float]], rows: list[list[int]],
                  stack_gap_ratio: float = 0.35) -> list[int]:
    """Associe chaque rangée à une étagère physique (rangées empilées = même étagère).

    Deux rangées consécutives sont sur la même étagère si l'écart vertical entre le bas
    de la rangée du dessus et le haut de la rangée du dessous est très faible (pas de
    planche entre les deux). Retourne l'index d'étagère (1-based) pour chaque rangée.
    """
    if not rows:
        return []
    med_h = float(np.median([b[3] - b[1] for b in boxes]))
    shelf_of_row = [1]
    for k in range(1, len(rows)):
        prev_bottom = np.median([boxes[i][3] for i in rows[k - 1]])
        cur_top = np.median([boxes[i][1] for i in rows[k]])
        same = (cur_top - prev_bottom) < stack_gap_ratio * med_h
        shelf_of_row.append(shelf_of_row[-1] if same else shelf_of_row[-1] + 1)
    return shelf_of_row


def find_gaps(boxes: list[list[float]], rows: list[list[int]],
              gap_factor: float = 1.0) -> list[dict]:
    """Trous horizontaux : écart x1_(i+1) - x2_i > gap_factor × largeur médiane."""
    if not boxes:
        return []
    med_w = float(np.median([b[2] - b[0] for b in boxes]))
    left = min(b[0] for b in boxes)
    right = max(b[2] for b in boxes)
    gaps: list[dict] = []
    for r_idx, row in enumerate(rows):
        y1 = float(np.median([boxes[i][1] for i in row]))
        y2 = float(np.median([boxes[i][3] for i in row]))
        xs = [(left, left)] + [(boxes[i][0], boxes[i][2]) for i in row] + [(right, right)]
        for (a1, a2), (b1, b2) in zip(xs[:-1], xs[1:]):
            if b1 - a2 > gap_factor * med_w:
                gaps.append({"bbox": [float(a2), y1, float(b1), y2], "row": r_idx + 1,
                             "largeur_en_boites": round((b1 - a2) / med_w, 1)})
    return gaps


def analyze_image(img: np.ndarray, method: str = "auto", weights: str | None = None,
                  conf: float = 0.25) -> dict:
    raw, used = detect_boxes(img, method, weights, conf)
    rows = group_rows(raw)
    shelves = group_shelves(raw, rows)
    row_of = {i: r for r, row in enumerate(rows) for i in row}
    items = [
        {"id": i, "bbox": [round(v, 1) for v in b[:4]], "score": round(b[4], 3),
         "row": row_of[i] + 1, "etagere": shelves[row_of[i]]}
        for i, b in enumerate(raw)
    ]
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "method": used,
        "boxes": items,
        "rows": [[int(i) for i in r] for r in rows],
        "gaps": find_gaps(raw, rows),
    }


def detect_items_and_gaps(image_path: str, method: str = "auto", weights: str | None = None,
                          conf: float = 0.25) -> dict:
    """Point d'entrée de la spec : détecte boîtes, rangées et trous sur une image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Image illisible : {image_path}")
    out = analyze_image(img, method, weights, conf)
    out["image"] = str(image_path)
    return out


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--detector", default="auto", choices=["auto", "yolo", "classical"])
    a = p.parse_args()
    r = detect_items_and_gaps(a.image, a.detector)
    print(json.dumps({k: v for k, v in r.items() if k != "boxes"}, indent=2, ensure_ascii=False))
    print(f"{len(r['boxes'])} boîtes détectées ({r['method']})")
