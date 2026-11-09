"""Module 3 — Comparaison « rayon plein » (référence) vs « rayon actuel ».

Étapes :
  1. Planogramme : sur la photo de référence, on détecte chaque boîte (= un emplacement)
     puis on regroupe les boîtes voisines visuellement identiques en « blocs produit »
     (une colonne / un facing d'un même médicament). Capacité du bloc = nb de boîtes.
     Le planogramme est sauvegardé à côté de la référence (``*.planogram.json``) et
     peut être corrigé à la main (noms de produits) depuis l'interface.
  2. Alignement : la capture du jour est recalée sur la référence par homographie
     SIFT + RANSAC (approche reprise de Retail-Shelf-Monitoring / shelf_aligner).
  3. Appariement : assignation bipartite hongroise (``linear_sum_assignment``) entre les
     emplacements de référence et les boîtes détectées, coût = (1 - IoU) + 0.5 × distance
     des centres normalisée (formule reprise de Shelf-Vision / planogram_matcher).
     Un contrôle d'apparence (histogramme HSV + corrélation normalisée) confirme la
     présence de la boîte, ce qui rend le système robuste aux ratés du détecteur.
  4. Différentiel par bloc : Δ = capacité_ref - quantité_actuelle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src import CROPS_DIR
from src.shelf_detector import analyze_image, box_iou

# Seuils de décision (similarité d'apparence entre -1 et 1)
SIM_PRESENT_STRONG = 0.55  # présent même sans détection appariée
SIM_PRESENT_WEAK = 0.25    # présent si une détection est appariée au même endroit
MIN_MATCH_IOU = 0.3
MIN_VISIBLE_FRACTION = 0.9


# --------------------------------------------------------------------------------------
# Appariement hongrois (adapté de Shelf-Vision, licence MIT)
# --------------------------------------------------------------------------------------
def _center_distance_norm(a, b, w: float, h: float) -> float:
    ax, ay = (a[0] + a[2]) / 2 / w, (a[1] + a[3]) / 2 / h
    bx, by = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
    return float(np.hypot(ax - bx, ay - by))


def hungarian_match(ref_boxes: list, cur_boxes: list, w: float, h: float,
                    min_iou: float = MIN_MATCH_IOU) -> dict[int, tuple[int, float]]:
    """Assignation optimale emplacements de référence → détections actuelles.

    Returns: {index_ref: (index_cur, iou)} pour les paires avec IoU >= min_iou.
    """
    if not ref_boxes or not cur_boxes:
        return {}
    cost = np.zeros((len(ref_boxes), len(cur_boxes)), dtype=np.float32)
    ious = np.zeros_like(cost)
    for i, rb in enumerate(ref_boxes):
        for j, cb in enumerate(cur_boxes):
            iou = box_iou(rb, cb)
            ious[i, j] = iou
            cost[i, j] = (1.0 - iou) + 0.5 * _center_distance_norm(rb, cb, w, h)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): (int(c), float(ious[r, c])) for r, c in zip(rows, cols) if ious[r, c] >= min_iou}


# --------------------------------------------------------------------------------------
# Similarité d'apparence
# --------------------------------------------------------------------------------------
def _crop(img: np.ndarray, box, margin: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    return img[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]


def look_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité visuelle de deux vignettes : moyenne histogramme HSV + NCC (−1…1)."""
    if a.size == 0 or b.size == 0:
        return -1.0
    size = (48, 64)
    a, b = cv2.resize(a, size), cv2.resize(b, size)
    hists = []
    for im in (a, b):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hists.append(cv2.normalize(hist, hist).flatten())
    h_sim = float(cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sa, sb = ga.std(), gb.std()
    if sa < 4 or sb < 4:  # zone uniforme (fond d'étagère vide)
        ncc = 0.0 if abs(sa - sb) < 2 and abs(ga.mean() - gb.mean()) < 15 else -0.5
    else:
        ncc = float(((ga - ga.mean()) * (gb - gb.mean())).mean() / (sa * sb))
    return 0.5 * h_sim + 0.5 * ncc


# --------------------------------------------------------------------------------------
# Planogramme (référence)
# --------------------------------------------------------------------------------------
def planogram_path(ref_image_path: str | Path) -> Path:
    p = Path(ref_image_path)
    return p.with_name(p.stem + ".planogram.json")


def build_planogram(ref_image_path: str, method: str = "auto", weights: str | None = None,
                    conf: float = 0.25, similarity_threshold: float = 0.6) -> dict:
    """Détecte les emplacements du rayon plein et les regroupe en blocs produit."""
    img = cv2.imread(str(ref_image_path))
    if img is None:
        raise IOError(f"Image de référence illisible : {ref_image_path}")
    det = analyze_image(img, method, weights, conf)
    boxes = [b["bbox"] for b in det["boxes"]]
    n = len(boxes)
    if n == 0:
        raise ValueError("Aucune boîte détectée sur la photo de référence.")

    med_w = float(np.median([b[2] - b[0] for b in boxes]))
    med_h = float(np.median([b[3] - b[1] for b in boxes]))
    crops = [_crop(img, b) for b in boxes]
    shelf = [b["etagere"] for b in det["boxes"]]

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def similar_size(a, b) -> bool:
        wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
        return 0.75 < wa / wb < 1.33 and 0.65 < ha / hb < 1.55

    for i in range(n):
        for j in range(i + 1, n):
            if shelf[i] != shelf[j] or not similar_size(boxes[i], boxes[j]):
                continue
            a, b = boxes[i], boxes[j]
            v_overlap = min(a[3], b[3]) - max(a[1], b[1])
            h_overlap = min(a[2], b[2]) - max(a[0], b[0])
            h_gap = max(a[0], b[0]) - min(a[2], b[2])
            v_gap = max(a[1], b[1]) - min(a[3], b[3])
            side_by_side = v_overlap > 0.5 * min(a[3] - a[1], b[3] - b[1]) and h_gap < 0.5 * med_w
            stacked = h_overlap > 0.6 * min(a[2] - a[0], b[2] - b[0]) and v_gap < 0.3 * med_h
            if not (side_by_side or stacked):
                continue
            # Compare des zones homologues : bande verticale commune pour deux boîtes
            # côte à côte (robuste à une détection tronquée), boîte entière sinon.
            if side_by_side:
                y1, y2 = max(a[1], b[1]), min(a[3], b[3])
                ca = _crop(img, [a[0], y1, a[2], y2])
                cb = _crop(img, [b[0], y1, b[2], y2])
            else:
                ca, cb = crops[i], crops[j]
            if look_similarity(ca, cb) >= similarity_threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    stem = Path(ref_image_path).stem
    blocks: list[dict] = []
    for members in groups.values():
        members.sort(key=lambda i: (boxes[i][1], boxes[i][0]))
        xs1, ys1 = [boxes[i][0] for i in members], [boxes[i][1] for i in members]
        xs2, ys2 = [boxes[i][2] for i in members], [boxes[i][3] for i in members]
        blocks.append({"slots": members, "etagere": shelf[members[0]],
                       "bbox": [min(xs1), min(ys1), max(xs2), max(ys2)]})
    blocks.sort(key=lambda b: (b["etagere"], b["bbox"][0], b["bbox"][1]))

    col_counter: dict[int, int] = {}
    out_blocks = []
    for k, b in enumerate(blocks):
        col_counter[b["etagere"]] = col_counter.get(b["etagere"], 0) + 1
        col = col_counter[b["etagere"]]
        # Vignette d'une boîte entière du bloc (pour l'identification VLM / catalogue)
        crop_path = CROPS_DIR / f"{stem}_ref_et{b['etagere']}_col{col}.jpg"
        cv2.imwrite(str(crop_path), _crop(img, boxes[b["slots"][0]], margin=0.03))
        out_blocks.append({
            "block_id": k + 1,
            "etagere": b["etagere"],
            "colonne": col,
            "produit": None,
            "produit_source": None,
            "capacite": len(b["slots"]),
            "slot_ids": b["slots"],
            "bbox": [round(v, 1) for v in b["bbox"]],
            "crop_ref": str(crop_path),
        })

    return {
        "reference_image": str(ref_image_path),
        "width": det["width"],
        "height": det["height"],
        "detector": det["method"],
        "similarity_threshold": similarity_threshold,
        "slots": [{"slot_id": i, "bbox": boxes[i], "etagere": shelf[i]} for i in range(n)],
        "blocks": out_blocks,
    }


def save_planogram(planogram: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else planogram_path(planogram["reference_image"])
    path.write_text(json.dumps(planogram, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_or_build_planogram(ref_image_path: str, rebuild: bool = False, **kwargs) -> dict:
    p = planogram_path(ref_image_path)
    if p.exists() and not rebuild and p.stat().st_mtime >= Path(ref_image_path).stat().st_mtime:
        return json.loads(p.read_text(encoding="utf-8"))
    planogram = build_planogram(ref_image_path, **kwargs)
    save_planogram(planogram, p)
    return planogram


# --------------------------------------------------------------------------------------
# Alignement (homographie SIFT + RANSAC, inspiré de Retail-Shelf-Monitoring)
# --------------------------------------------------------------------------------------
def _scale_matrix(s: float) -> np.ndarray:
    return np.diag([s, s, 1.0])


def _homography_is_sane(H: np.ndarray) -> bool:
    det = np.linalg.det(H[:2, :2])
    if det <= 0.02 or det > 50:
        return False
    if abs(H[2, 0]) > 0.005 or abs(H[2, 1]) > 0.005:
        return False
    s = np.linalg.svd(H[:2, :2], compute_uv=False)
    return s[0] / s[1] < 10


def align_to_reference(ref_img: np.ndarray, cur_img: np.ndarray,
                       max_side: int = 1600) -> dict:
    """Homographie qui projette la capture courante dans le repère de la référence."""
    def prep(img):
        s = min(1.0, max_side / max(img.shape[:2]))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g), s

    g_ref, s_ref = prep(ref_img)
    g_cur, s_cur = prep(cur_img)
    sift = cv2.SIFT_create(nfeatures=6000)
    kr, dr = sift.detectAndCompute(g_ref, None)
    kc, dc = sift.detectAndCompute(g_cur, None)

    H, inliers = None, 0
    if dr is not None and dc is not None and len(kr) >= 10 and len(kc) >= 10:
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
        good = [m for m, n in (p for p in matcher.knnMatch(dc, dr, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) >= 12:
            src = np.float32([kc[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=3000)
            if Hs is not None:
                inliers = int(mask.sum())
                H_full = np.linalg.inv(_scale_matrix(s_ref)) @ Hs @ _scale_matrix(s_cur)
                H_full /= H_full[2, 2]
                if inliers >= 15 and _homography_is_sane(H_full):
                    H = H_full

    if H is not None:
        return {"H": H, "inliers": inliers, "method": "sift-homography"}

    # Repli : même cadrage supposé, simple mise à l'échelle.
    rh, rw = ref_img.shape[:2]
    ch, cw = cur_img.shape[:2]
    if abs(rw / rh - cw / ch) < 0.05 * (rw / rh):
        H = np.diag([rw / cw, rh / ch, 1.0])
        return {"H": H, "inliers": inliers, "method": "identity-fallback"}
    return {"H": None, "inliers": inliers, "method": "failed"}


def _transform_box(box, H: np.ndarray) -> list[float]:
    pts = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
    t = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return [float(t[:, 0].min()), float(t[:, 1].min()), float(t[:, 0].max()), float(t[:, 1].max())]


# --------------------------------------------------------------------------------------
# Comparaison
# --------------------------------------------------------------------------------------
def evaluate_frame(ref_img: np.ndarray, cur_img: np.ndarray, planogram: dict,
                   method: str = "auto", weights: str | None = None, conf: float = 0.25) -> dict:
    """Évalue la présence de chaque emplacement de référence sur une capture."""
    W, Hh = planogram["width"], planogram["height"]
    if ref_img.shape[1] != W or ref_img.shape[0] != Hh:
        ref_img = cv2.resize(ref_img, (W, Hh))
    al = align_to_reference(ref_img, cur_img)
    if al["H"] is None:
        return {"alignment": al, "slots": {}, "warped": None, "detections": [], "gaps": []}
    H = al["H"]
    warped = cv2.warpPerspective(cur_img, H, (W, Hh), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(cur_img.shape[:2], 255, np.uint8), H, (W, Hh),
                                flags=cv2.INTER_NEAREST) > 0

    det = analyze_image(cur_img, method, weights, conf)
    cur_boxes = [_transform_box(b["bbox"], H) for b in det["boxes"]]
    gaps = [{**g, "bbox": _transform_box(g["bbox"], H)} for g in det["gaps"]]
    ref_boxes = [s["bbox"] for s in planogram["slots"]]
    matches = hungarian_match(ref_boxes, cur_boxes, W, Hh)

    slots: dict[int, dict] = {}
    for s in planogram["slots"]:
        sid = s["slot_id"]
        x1, y1, x2, y2 = (int(round(v)) for v in s["bbox"])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(Hh, y2)
        if x2 <= x1 or y2 <= y1 or valid[y1:y2, x1:x2].mean() < MIN_VISIBLE_FRACTION:
            continue
        sim = look_similarity(_crop(ref_img, s["bbox"]), _crop(warped, s["bbox"]))
        m = matches.get(sid)
        present = sim >= SIM_PRESENT_STRONG or (m is not None and sim >= SIM_PRESENT_WEAK)
        slots[sid] = {"present": bool(present), "similarity": round(sim, 3),
                      "matched_iou": round(m[1], 3) if m else 0.0}
    return {"alignment": al, "slots": slots, "warped": warped, "detections": cur_boxes,
            "gaps": gaps}


def compare_with_reference(ref_image_path: str, current_image_path: str | Iterable[str],
                           planogram: dict | None = None, method: str = "auto",
                           weights: str | None = None, conf: float = 0.25,
                           save_crops: bool = True, return_details: bool = False):
    """Compare une ou plusieurs captures au rayon plein.

    Args:
        ref_image_path: photo du rayon plein.
        current_image_path: une image ou une liste de frames (vidéo).
        planogram: planogramme pré-calculé (sinon chargé / construit automatiquement).

    Returns:
        Liste des anomalies par bloc produit (ou ``(anomalies, details)`` si
        ``return_details``). Chaque anomalie contient les coordonnées de découpe
        (repère de la référence) des boîtes manquantes.
    """
    frames = [current_image_path] if isinstance(current_image_path, (str, Path)) else list(current_image_path)
    if planogram is None:
        planogram = load_or_build_planogram(ref_image_path, method=method, weights=weights, conf=conf)
    ref_img = cv2.imread(str(ref_image_path))
    if ref_img is None:
        raise IOError(f"Référence illisible : {ref_image_path}")

    votes: dict[int, list[bool]] = {}
    frame_results: list[dict[str, Any]] = []
    for f in frames:
        cur = cv2.imread(str(f))
        if cur is None:
            continue
        r = evaluate_frame(ref_img, cur, planogram, method, weights, conf)
        r["path"] = str(f)
        frame_results.append(r)
        for sid, st in r["slots"].items():
            votes.setdefault(sid, []).append(st["present"])

    slot_state: dict[int, str] = {}
    for s in planogram["slots"]:
        v = votes.get(s["slot_id"])
        if not v:
            slot_state[s["slot_id"]] = "non_vu"
        else:
            slot_state[s["slot_id"]] = "present" if sum(v) / len(v) >= 0.5 else "absent"

    slot_by_id = {s["slot_id"]: s for s in planogram["slots"]}
    anomalies: list[dict] = []
    for b in planogram["blocks"]:
        states = [slot_state[i] for i in b["slot_ids"]]
        missing = [i for i in b["slot_ids"] if slot_state[i] == "absent"]
        unseen = states.count("non_vu")
        if not missing:
            continue
        cap = b["capacite"]
        present = cap - len(missing)  # emplacements non vus supposés présents
        missing_boxes = [slot_by_id[i]["bbox"] for i in missing]
        u = [min(x[0] for x in missing_boxes), min(x[1] for x in missing_boxes),
             max(x[2] for x in missing_boxes), max(x[3] for x in missing_boxes)]

        proof = None
        if save_crops:
            # frame qui voit le plus d'emplacements manquants de ce bloc
            best = max((fr for fr in frame_results if fr["warped"] is not None),
                       key=lambda fr: sum(i in fr["slots"] for i in missing), default=None)
            if best is not None:
                proof_path = CROPS_DIR / f"trou_et{b['etagere']}_col{b['colonne']}.jpg"
                cv2.imwrite(str(proof_path), _crop(best["warped"], u, margin=0.25))
                proof = str(proof_path)

        anomalies.append({
            "block_id": b["block_id"],
            "produit": b.get("produit"),
            "etagere": b["etagere"],
            "colonne": b["colonne"],
            "type_anomalie": "Rupture" if present == 0 else "Colonne incomplète",
            "quantite_presente": present,
            "capacite_totale": cap,
            "quantite_a_reassortir": len(missing),
            "emplacements_non_vus": unseen,
            "bbox_bloc": b["bbox"],
            "bbox_manquants": [[round(v, 1) for v in x] for x in missing_boxes],
            "bbox_decoupe": [round(v, 1) for v in u],
            "crop_ref": b.get("crop_ref"),
            "image_preuve": proof,
        })

    if not return_details:
        return anomalies
    details = {
        "planogram": planogram,
        "slot_state": slot_state,
        "frames": [{"path": fr["path"], "alignment": fr["alignment"]["method"],
                    "inliers": fr["alignment"]["inliers"], "slots_vus": len(fr["slots"])}
                   for fr in frame_results],
        "frame_results": frame_results,
    }
    return anomalies, details


# --------------------------------------------------------------------------------------
# Annotation
# --------------------------------------------------------------------------------------
RED, ORANGE, GREEN, GREY = (40, 40, 230), (0, 140, 255), (60, 180, 60), (150, 150, 150)


def annotate(image: np.ndarray, planogram: dict, anomalies: list[dict],
             H_ref_to_img: np.ndarray | None = None, gaps: list[dict] | None = None) -> np.ndarray:
    """Dessine les zones à remplir : rouge = rupture, orange = colonne incomplète."""
    out = image.copy()
    thick = max(2, int(round(max(out.shape[:2]) / 400)))
    font_scale = max(0.5, max(out.shape[:2]) / 1600)

    def proj(box):
        return _transform_box(box, H_ref_to_img) if H_ref_to_img is not None else box

    overlay = out.copy()
    for a in anomalies:
        color = RED if a["type_anomalie"] == "Rupture" else ORANGE
        for mb in a["bbox_manquants"]:
            x1, y1, x2, y2 = (int(v) for v in proj(mb))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    for g in gaps or []:
        x1, y1, x2, y2 = (int(v) for v in proj(g["bbox"]))
        cv2.rectangle(out, (x1, y1), (x2, y2), GREY, max(1, thick // 2))

    for a in anomalies:
        color = RED if a["type_anomalie"] == "Rupture" else ORANGE
        x1, y1, x2, y2 = (int(v) for v in proj(a["bbox_bloc"]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"#{a.get('id', a['block_id'])} -{a['quantite_a_reassortir']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return out


def annotate_planogram(ref_img: np.ndarray, planogram: dict) -> np.ndarray:
    """Visualise les blocs produit détectés sur le rayon plein (paramétrage)."""
    out = ref_img.copy()
    thick = max(2, int(round(max(out.shape[:2]) / 500)))
    fs = max(0.45, max(out.shape[:2]) / 1800)
    for s in planogram["slots"]:
        x1, y1, x2, y2 = (int(v) for v in s["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), GREY, 1)
    for b in planogram["blocks"]:
        x1, y1, x2, y2 = (int(v) for v in b["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, thick)
        label = f"E{b['etagere']}C{b['colonne']} x{b['capacite']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 6, y1 + th + 8), GREEN, -1)
        cv2.putText(out, label, (x1 + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
