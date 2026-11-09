"""Contrôle « vide / occupé » de l'emplacement situé au-dessus de chaque étiquette.

Pour chaque étiquette de prix, on examine la zone de la tablette juste au-dessus
(là où le produit étiqueté doit se trouver) et on combine trois indices :
  * YOLO voit-il un produit dans la zone ?            (flacons surtout)
  * densité de contours (texte, logos, bords d'objets) (tous types de produits)
  * proportion de pixels colorés (emballages)          (tous types de produits)
Un emplacement n'est déclaré VIDE que si aucun de ces indices ne signale un objet :
on préfère rater un trou plutôt que d'envoyer quelqu'un remplir un rayon déjà plein.

Ensuite, pour les emplacements OCCUPÉS, une carte de profondeur (src/depth_estimator.py)
indique si, entre l'avant de la tablette (l'étiquette) et le mur du fond :
  * il y a un ESPACE : sur une bonne partie de la largeur on voit loin derrière
    (le mur) → un produit manque à côté des autres ;
  * le produit est EN RETRAIT : la façade est loin derrière l'étiquette → les boîtes de
    devant ont été vendues, il reste seulement celles du fond.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.label_detector import detect_labels

# Seuils calibrés sur la vidéo réelle « video teste » (cœur de zone) :
#   produits présents : contours ≥ 0,055 dans 95 % des cas (médiane 0,156)
#   emplacements vides : contours ≤ 0,015
# À affiner avec une vidéo contenant de vrais trous connus.
EDGE_OCCUPIED = 0.035   # fraction de pixels « contour » au-delà de laquelle c'est occupé
COLOR_OCCUPIED = 0.25   # fraction de pixels nettement colorés
# Une détection YOLO faible (< 0,35) sur une zone sans aucun contour est presque toujours
# un « fantôme » (ombre, fond uni) : elle ne suffit pas à déclarer l'emplacement occupé.
YOLO_CONF_SURE = 0.35
EDGE_MIN_WITH_WEAK_YOLO = 0.01
# Zone parfaitement uniforme (aucun contour) : il faut que YOLO soit TRÈS sûr de lui
# pour contredire l'image (un fond uni rectangulaire peut ressembler à une boîte).
EDGE_UNIFORM = 0.005
YOLO_CONF_UNIFORM = 0.7

# Profondeur : « recul » = (disparité étiquette − disparité zone) / disparité étiquette.
#   0 = au ras de l'étiquette (avant du rayon), grand = loin derrière (vers le mur).
# Calibré sur « video teste » (232 emplacements occupés) :
#   recul médian de la façade : 0,07 (98 % ≤ 0,26) ; emplacements vides : 0,23 à 0,66.
#   plus longue bande « profonde » (> 0,3) : 0 en médiane (98 % ≤ 0,41).
DEPTH_DEEP = 0.30        # une colonne est « profonde » au-delà de ce recul
SPACE_RUN = 0.45         # espace : bande profonde continue sur ≥ 45 % de la largeur
RECESS_MEDIAN = 0.30     # retrait : la façade entière est reculée d'au moins 0,30
# Pile incomplète : des produits sont bien à l'avant, mais au-dessus d'eux on voit le mur
# sur une grande partie de la hauteur de la tablette (ex. 2 petites boîtes là où il en
# faudrait une pile). Lignes « profondes » : recul médian de la ligne > 0,40.
# Calibré sur « video teste » : 98 % des emplacements occupés ≤ 0,27 ; tubes debout plus
# courts que la tablette : 0,26 à 0,46 ; pile incomplète signalée par l'utilisatrice : 0,63.
ROW_DEEP = 0.40
INCOMPLETE_ROWS = 0.50


def label_zone(label: dict, labels: list[dict], img_shape) -> list[float]:
    """Zone produit au-dessus d'une étiquette (repère image)."""
    H, W = img_shape[:2]
    x1, y1, x2, y2 = label["bbox"]
    lw, lh = x2 - x1, y2 - y1
    bottom = y1 - 0.9 * lh               # on saute le bandeau de catégorie
    top = bottom - 3.5 * lh
    # ne pas déborder sur la rangée d'étiquettes du dessus
    above = [l["bbox"][3] for l in labels
             if l["bbox"][3] < y1 - lh and min(l["bbox"][2], x2 + lw) > max(l["bbox"][0], x1 - lw)]
    if above:
        top = max(top, max(above) + 0.2 * lh)
    zx1, zx2 = x1 - 0.25 * lw, x2 + 0.25 * lw
    return [max(0.0, zx1), max(0.0, top), min(float(W), zx2), max(0.0, bottom)]


def zone_core(zone: list[float]) -> list[float]:
    """Cœur de la zone : on ignore les bords, où débordent souvent les produits voisins."""
    x1, y1, x2, y2 = zone
    dx, dy = 0.22 * (x2 - x1), 0.12 * (y2 - y1)
    return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]


def zone_features(img: np.ndarray, zone: list[float], yolo_boxes: list[list[float]] | None) -> dict:
    """yolo_boxes : [x1, y1, x2, y2] ou [x1, y1, x2, y2, confiance]."""
    x1, y1, x2, y2 = (int(round(v)) for v in zone_core(zone))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return {"valide": False}
    patch = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    colored = (hsv[..., 1] > 70) & (hsv[..., 2] > 60)
    # YOLO ne compte que s'il voit un objet DANS la zone : centre horizontal dans le cœur,
    # et l'objet recouvre une bonne partie du cœur. Un produit de l'étagère du dessus,
    # du dessous ou un voisin qui déborde ne suffit pas.
    yolo_conf = 0.0
    core_area = float((x2 - x1) * (y2 - y1))
    for b in yolo_boxes or []:
        cx = (b[0] + b[2]) / 2
        ix = max(0.0, min(b[2], x2) - max(b[0], x1))
        iy = max(0.0, min(b[3], y2) - max(b[1], y1))
        if x1 <= cx <= x2 and ix * iy >= 0.4 * core_area and (b[2] - b[0]) < 2.5 * (x2 - x1):
            conf = float(b[4]) if len(b) > 4 else 1.0
            yolo_conf = max(yolo_conf, conf)
    contours = round(float((edges > 0).mean()), 3)
    if contours < EDGE_UNIFORM:
        yolo = yolo_conf >= YOLO_CONF_UNIFORM
    else:
        yolo = yolo_conf >= YOLO_CONF_SURE or (yolo_conf > 0 and contours >= EDGE_MIN_WITH_WEAK_YOLO)
    return {"valide": True, "contours": contours, "couleur": round(float(colored.mean()), 3),
            "yolo": bool(yolo), "yolo_conf": round(yolo_conf, 2)}


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def depth_features(depth: np.ndarray, label: dict, zone: list[float]) -> dict:
    """Recul de la façade et plus grand espace profond, mesurés dans le bas de la zone
    (là où se trouve l'avant des produits posés sur la tablette)."""
    lx1, ly1, lx2, ly2 = (int(round(v)) for v in label["bbox"])
    zx1, zy1, zx2, zy2 = (int(round(v)) for v in zone)
    h, w = zy2 - zy1, zx2 - zx1
    if h < 10 or w < 10 or lx2 <= lx1 or ly2 <= ly1:
        return {}
    d_lab = float(np.median(depth[ly1:ly2, lx1:lx2]))
    if d_lab <= 1e-6:
        return {}
    band = depth[zy1 + int(0.55 * h):zy1 + int(0.9 * h), zx1 + int(0.08 * w):zx2 - int(0.08 * w)]
    if band.size == 0:
        return {}
    recul_col = (d_lab - np.median(band, axis=0)) / d_lab
    n = len(recul_col)
    centre = recul_col[int(0.2 * n):max(int(0.8 * n), int(0.2 * n) + 1)]
    # profil vertical sur toute la hauteur de la zone (colonnes centrales)
    full = depth[zy1:zy2, zx1 + int(0.2 * w):zx2 - int(0.2 * w)]
    recul_row = (d_lab - np.median(full, axis=1)) / d_lab
    return {"recul": round(float(np.median(centre)), 2),
            "espace": round(_longest_run(recul_col > DEPTH_DEEP) / n, 2),
            "hauteur_vide": round(float((recul_row > ROW_DEEP).mean()), 2)}


def classify(f: dict) -> str:
    if not f.get("valide"):
        return "non_evalue"
    if not (f["contours"] >= EDGE_OCCUPIED or f["couleur"] >= COLOR_OCCUPIED or f["yolo"]):
        return "vide"
    if f.get("espace", 0.0) >= SPACE_RUN:
        return "espace"
    if f.get("hauteur_vide", 0.0) >= INCOMPLETE_ROWS:
        return "incomplet"
    if f.get("recul", 0.0) >= RECESS_MEDIAN:
        return "retrait"
    return "occupe"


def check_frame(img: np.ndarray, yolo_boxes: list[list[float]] | None = None,
                depth: np.ndarray | None = None, labels: list[dict] | None = None) -> list[dict]:
    """Étiquettes détectées + état de l'emplacement au-dessus de chacune.

    etat : "vide" | "espace" | "incomplet" | "retrait" (les trois derniers seulement si
           `depth` est fourni)
           | "occupe" | "non_evalue".
    """
    if labels is None:
        labels = detect_labels(img)
    out = []
    for l in labels:
        zone = label_zone(l, labels, img.shape)
        f = zone_features(img, zone, yolo_boxes)
        if depth is not None and f.get("valide"):
            f.update(depth_features(depth, l, zone))
        out.append({**l, "zone": [round(v, 1) for v in zone], "mesures": f, "etat": classify(f)})
    return out


# rouge = vide, orange = espace entre produits, bleu = pile incomplète,
# jaune = produit en retrait (BGR)
ANOMALY_COLORS = {"vide": (0, 0, 255), "espace": (0, 140, 255), "incomplet": (255, 120, 0),
                  "retrait": (0, 215, 255)}


def draw(img: np.ndarray, slots: list[dict]) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    for s in slots:
        x1, y1, x2, y2 = (int(v) for v in s["zone"])
        if s["etat"] in ANOMALY_COLORS:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), ANOMALY_COLORS[s["etat"]], -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    for s in slots:
        lx1, ly1, lx2, ly2 = (int(v) for v in s["bbox"])
        x1, y1, x2, y2 = (int(v) for v in s["zone"])
        color = {**ANOMALY_COLORS, "occupe": (0, 190, 0)}.get(s["etat"], (160, 160, 160))
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (255, 0, 255), 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if s["etat"] in ANOMALY_COLORS else 1)
    return out
