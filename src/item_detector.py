"""Détection de CHAQUE produit visible du rayon (boîtes, flacons, tubes…), 100 % local.

Deux détecteurs complémentaires, fusionnés :
  * Grounding DINO tiny (IDEA-Research, Apache-2.0) : détecteur « à vocabulaire ouvert »,
    on lui demande en texte « une boîte. un flacon. un tube. » → il trouve aussi les boîtes,
    ce que les modèles COCO (YOLO, RT-DETR) ne savent pas faire (« boîte » n'est pas une de
    leurs 80 classes).
  * YOLO / RT-DETR (COCO) : très bons sur les flacons.
Les doublons sont supprimés (NMS), puis on écarte ce qui n'est pas un produit : étiquettes
de prix, très grands cadres qui englobent toute une rangée, bandeaux, etc.

Chaque produit retenu est numéroté et dessiné : c'est la preuve qu'il a été vu.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

from src import MODELS_DIR

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_PROMPT = "a box. a bottle. a tube. a pump bottle. a carton package."
GDINO_BOX_THR = 0.18
GDINO_TEXT_THR = 0.18
# Découpage en tuiles (2 colonnes × 3 lignes, ~30 % de recouvrement) : chaque tuile est
# agrandie par le modèle → les petites boîtes deviennent visibles. Sur « video teste »
# (image 0:09) : 25 produits trouvés sur l'image entière, 51 avec les tuiles.
TILE_COLS, TILE_ROWS = 2, 3
TILE_W, TILE_H = 0.62, 0.45
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@lru_cache(maxsize=1)
def _gdino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(GDINO_ID, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID, local_files_only=True)
    except OSError:
        proc = AutoProcessor.from_pretrained(GDINO_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID)
    _fix_fused_attention(model)
    return proc, model.eval()


def _fix_fused_attention(model) -> None:
    """Le fichier publié range les projections d'attention « fusionnées » (in_proj_weight,
    3 matrices collées) alors que transformers ≥ 5 attend query / key / value séparées :
    sans cette conversion ces couches restent ALÉATOIRES et le modèle ne détecte rien."""
    import torch
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(GDINO_ID, "pytorch_model.bin", local_files_only=True)
    except Exception:
        path = hf_hub_download(GDINO_ID, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    fixed = {}
    for k, v in sd.items():
        if k.endswith(("in_proj_weight", "in_proj_bias")):
            base, kind = k.rsplit(".", 1)
            suffix = "weight" if kind == "in_proj_weight" else "bias"
            q, kk, vv = v.chunk(3, dim=0)
            fixed[f"{base}.query.{suffix}"] = q
            fixed[f"{base}.key.{suffix}"] = kk
            fixed[f"{base}.value.{suffix}"] = vv
        elif k.endswith(("out_proj.weight", "out_proj.bias")):
            fixed[k] = v
    own = model.state_dict()
    usable = {k: v for k, v in fixed.items() if k in own and own[k].shape == v.shape}
    # out_proj : nom identique ou « out_proj » → vérifier qu'il existe bien dans le modèle
    model.load_state_dict(usable, strict=False)


def detect_gdino(img: np.ndarray) -> list[list[float]]:
    import torch
    from PIL import Image

    proc, model = _gdino()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = proc(images=pil, text=GDINO_PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    r = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=GDINO_BOX_THR, text_threshold=GDINO_TEXT_THR,
        target_sizes=[img.shape[:2]])[0]
    return [[*map(float, b.tolist()), float(s)] for b, s in zip(r["boxes"], r["scores"])]


def detect_gdino_tiled(img: np.ndarray) -> list[list[float]]:
    """Grounding DINO sur l'image entière + sur des tuiles qui se recouvrent."""
    H, W = img.shape[:2]
    tw, th = int(W * TILE_W), int(H * TILE_H)
    out = detect_gdino(img)
    for y in np.linspace(0, H - th, TILE_ROWS).astype(int):
        for x in np.linspace(0, W - tw, TILE_COLS).astype(int):
            for b in detect_gdino(img[y:y + th, x:x + tw]):
                bb = [b[0] + x, b[1] + y, b[2] + x, b[3] + y, b[4]]
                # objet coupé par le bord intérieur d'une tuile : il sera vu entier ailleurs
                if (b[0] < 2 and x > 0) or (b[2] > tw - 2 and x + tw < W)                         or (b[1] < 2 and y > 0) or (b[3] > th - 2 and y + th < H):
                    continue
                out.append(bb)
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _contain(a, b) -> float:
    """Part de `a` contenue dans `b`."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def _nms(boxes: list[list[float]], thr: float = 0.5) -> list[list[float]]:
    keep: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def clean_items(boxes: list[list[float]], img_shape, labels: list[dict] | None = None) -> list[list[float]]:
    """Retire ce qui n'est pas un produit individuel."""
    H, W = img_shape[:2]
    out = []
    for b in boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= 2 or h <= 2:
            continue
        area = w * h / (W * H)
        if area < 0.0015 or area > 0.12:           # miette ou cadre géant
            continue
        if w > 0.45 * W or w / h > 4.0:             # bandeau / rangée entière
            continue
        if labels and any(_contain(b, l["bbox"]) > 0.5 or _contain(l["bbox"], b) > 0.8 and area < 0.01
                          for l in labels):         # c'est une étiquette de prix
            continue
        out.append(b)
    # un cadre qui contient ≥ 2 produits déjà trouvés = groupe, pas un produit
    final = []
    for b in out:
        inside = [o for o in out if o is not b and _contain(o, b) > 0.85
                  and (o[2] - o[0]) * (o[3] - o[1]) < 0.7 * (b[2] - b[0]) * (b[3] - b[1])]
        if len(inside) >= 2:
            continue
        final.append(b)
    return final


def detect_items(img: np.ndarray, coco_boxes: list[list[float]] | None = None,
                 labels: list[dict] | None = None, use_gdino: bool = True,
                 tiled: bool = True) -> list[dict]:
    """Tous les produits visibles : [{"id", "bbox", "score", "source"}] triés haut→bas, gauche→droite."""
    cand: list[list[float]] = []
    sources: dict[int, str] = {}
    if use_gdino:
        for b in (detect_gdino_tiled(img) if tiled else detect_gdino(img)):
            cand.append(b[:5])
            sources[id(cand[-1])] = "grounding-dino"
    for b in coco_boxes or []:
        bb = [float(v) for v in b[:4]] + [float(b[4]) if len(b) > 4 else 0.5]
        cand.append(bb)
        sources[id(cand[-1])] = "yolo"
    kept = clean_items(_nms(cand, 0.45), img.shape, labels)
    kept.sort(key=lambda b: (round(((b[1] + b[3]) / 2) / 40), b[0]))
    return [{"id": i + 1, "bbox": [round(v, 1) for v in b[:4]], "score": round(b[4], 2),
             "source": sources.get(id(b), "?")} for i, b in enumerate(kept)]


def draw_items(img: np.ndarray, items: list[dict], color=(0, 220, 0)) -> np.ndarray:
    """Preuve visuelle : chaque produit détecté encadré et numéroté."""
    out = img.copy()
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = str(it["id"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 4), color, -1)
        cv2.putText(out, txt, (x1 + 2, y1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(out, f"{len(items)} produits detectes", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 3)
    cv2.putText(out, f"{len(items)} produits detectes", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 200), 2)
    return out
