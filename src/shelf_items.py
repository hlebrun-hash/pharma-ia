"""Relier chaque produit détecté à sa tablette / à son étiquette, et trouver les espaces
ENTRE deux produits voisins (un produit manque entre eux).

Repères : les rangées d'étiquettes de prix sont l'avant des tablettes. Un produit appartient
à la tablette dont la rangée d'étiquettes est juste en dessous de lui.
"""

from __future__ import annotations

import cv2
import numpy as np

GAP_MIN_REL = 0.6        # espace ≥ 0,6 × largeur médiane des produits de la tablette
GAP_MIN_LABEL = 0.6      # … et ≥ 0,6 × largeur d'une étiquette
GAP_DEEP = 0.30          # profondeur : on voit loin derrière (le mur) dans l'espace
GAP_EDGES_MAX = 0.035    # sans profondeur : l'espace doit être visuellement vide


def _rows(slots: list[dict]) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = {}
    for s in slots:
        rows.setdefault(s["rangee"], []).append(s)
    return rows


def assign_items(items: list[dict], slots: list[dict]) -> dict[int, list[dict]]:
    """Ajoute à chaque emplacement la liste des produits détectés au-dessus de son étiquette
    (slot["produits"]) et renvoie {rangée: produits de cette tablette}."""
    rows = _rows(slots)
    order = sorted(rows, key=lambda r: np.median([s["bbox"][1] for s in rows[r]]))
    tops = {r: float(np.median([s["bbox"][1] for s in rows[r]])) for r in order}
    bottoms = {r: float(np.median([s["bbox"][3] for s in rows[r]])) for r in order}
    lh = float(np.median([s["bbox"][3] - s["bbox"][1] for s in slots])) if slots else 0.0
    per_row: dict[int, list[dict]] = {r: [] for r in order}
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        cy = (y1 + y2) / 2
        # la tablette = première rangée d'étiquettes sous le centre du produit
        below = [r for r in order if tops[r] > cy - 0.2 * lh]
        if not below:
            continue
        r = below[0]
        prev = order[order.index(r) - 1] if order.index(r) > 0 else None
        if prev is not None and cy < bottoms[prev]:
            continue
        if y2 > tops[r] + 1.0 * lh:      # descend sous les étiquettes : pas cette tablette
            continue
        it["rangee"] = r
        per_row[r].append(it)
    for s in slots:
        zx1, _, zx2, _ = s["zone"]
        s["produits"] = [it["id"] for it in per_row.get(s["rangee"], [])
                         if zx1 <= (it["bbox"][0] + it["bbox"][2]) / 2 <= zx2]
    return per_row


def _gap_is_empty(img, depth, gap_box, label_bbox) -> tuple[bool, dict]:
    x1, y1, x2, y2 = (int(round(v)) for v in gap_box)
    h = y2 - y1
    if x2 - x1 < 3 or h < 6:
        return False, {}
    core = img[y1 + int(0.15 * h):y2 - int(0.15 * h), x1:x2]
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    edges = float((cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120) > 0).mean())
    m = {"contours": round(edges, 3)}
    if depth is not None:
        lx1, ly1, lx2, ly2 = (int(round(v)) for v in label_bbox)
        d_lab = float(np.median(depth[ly1:ly2, lx1:lx2]))
        band = depth[y1 + int(0.3 * h):y2 - int(0.1 * h), x1:x2]
        if d_lab > 0 and band.size:
            recul = (d_lab - float(np.median(band))) / d_lab
            m["recul"] = round(recul, 2)
            return recul >= GAP_DEEP, m
    return edges < GAP_EDGES_MAX, m


def find_item_gaps(img: np.ndarray, items_per_row: dict[int, list[dict]], slots: list[dict],
                   depth: np.ndarray | None = None) -> list[dict]:
    """Espaces entre deux produits voisins d'une même tablette."""
    rows = _rows(slots)
    gaps = []
    for r, its in items_per_row.items():
        if len(its) < 2 or r not in rows:
            continue
        its = sorted(its, key=lambda i: i["bbox"][0])
        widths = [i["bbox"][2] - i["bbox"][0] for i in its]
        med_w = float(np.median(widths))
        lw = float(np.median([s["bbox"][2] - s["bbox"][0] for s in rows[r]]))
        shelf_top = float(np.median([i["bbox"][1] for i in its]))
        shelf_bottom = float(np.median([s["bbox"][1] for s in rows[r]]))
        # bord droit atteint jusqu'ici (un produit large peut en recouvrir plusieurs)
        reach_x, reach_item = its[0]["bbox"][2], its[0]
        for nxt in its[1:]:
            gap = nxt["bbox"][0] - reach_x
            if gap >= max(GAP_MIN_REL * med_w, GAP_MIN_LABEL * lw):
                y_bottom = min(shelf_bottom, max(reach_item["bbox"][3], nxt["bbox"][3]))
                y_top = min(shelf_top, y_bottom - 20)
                box = [reach_x + 2, max(0.0, y_top), nxt["bbox"][0] - 2, y_bottom]
                cx = (box[0] + box[2]) / 2
                label = min(rows[r], key=lambda s: abs((s["bbox"][0] + s["bbox"][2]) / 2 - cx))
                empty, m = _gap_is_empty(img, depth, box, label["bbox"])
                if empty:
                    gaps.append({"bbox": label["bbox"], "screen": label.get("screen"),
                                 "rangee": r, "index": label["index"],
                                 "zone": [round(v, 1) for v in box], "etat": "espace",
                                 "mesures": {**m, "entre_produits": [reach_item["id"], nxt["id"]]},
                                 "source": "produits"})
            if nxt["bbox"][2] > reach_x:
                reach_x, reach_item = nxt["bbox"][2], nxt
    return gaps


def _overlap_min(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / m if m > 0 else 0.0


def new_gaps_only(gaps: list[dict], anomalies: list[dict]) -> list[dict]:
    """Garde les espaces entre produits qui ne recoupent pas une anomalie déjà trouvée."""
    out = []
    for g in gaps:
        if any(_overlap_min(g["zone"], a["zone"]) > 0.3 for a in anomalies + out):
            continue
        out.append(g)
    return out


ITEM_COLOR = (255, 200, 0)       # bleu clair (BGR)
LABEL_COLOR = (255, 0, 255)      # violet
STATE_COLORS = {"vide": (0, 0, 255), "espace": (0, 140, 255), "incomplet": (255, 120, 0),
                "retrait": (0, 215, 255)}


def draw_proof(img: np.ndarray, items: list[dict], slots: list[dict], anomalies: list[dict],
               wall: np.ndarray | None = None) -> np.ndarray:
    """Image de preuve : chaque produit détecté (numéroté), chaque étiquette avec le nombre
    de produits vus au-dessus, le mur visible (vert translucide) et les anomalies en couleur."""
    if wall is not None:
        from src.wall_map import tint_wall
        img = tint_wall(img, wall)
    out = img.copy()
    overlay = out.copy()
    for a in anomalies:
        x1, y1, x2, y2 = (int(v) for v in a["zone"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), STATE_COLORS.get(a["etat"], (0, 0, 255)), -1)
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    for a in anomalies:
        x1, y1, x2, y2 = (int(v) for v in a["zone"])
        cv2.rectangle(out, (x1, y1), (x2, y2), STATE_COLORS.get(a["etat"], (0, 0, 255)), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), ITEM_COLOR, 2)
        txt = str(it["id"])
        if it.get("nom_lu"):  # nom lu sur la boîte (OCR), tronqué pour rester lisible
            name = "".join(c for c in it["nom_lu"] if ord(c) < 128)[:14]
            txt += f" {name}" if name else ""
        (tw, th), _ = cv2.getTextSize(txt, font, 0.42, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 4), ITEM_COLOR, -1)
        cv2.putText(out, txt, (x1 + 2, y1 + th + 1), font, 0.42, (0, 0, 0), 1)
    for s in slots:
        x1, y1, x2, y2 = (int(v) for v in s["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), LABEL_COLOR, 2)
        if "produits" in s:
            txt = str(len(s["produits"]))
            cv2.putText(out, txt, (x1 + 2, y2 + 14), font, 0.5, (255, 255, 255), 3)
            cv2.putText(out, txt, (x1 + 2, y2 + 14), font, 0.5, LABEL_COLOR, 1)
    head = f"{len(items)} produits detectes - {len(slots)} etiquettes - {len(anomalies)} anomalie(s)"
    if wall is not None:
        head += " - vert = mur visible"
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, head, (6, 21), font, 0.55, (255, 255, 255), 1)
    return out
