"""Détection des étiquettes de prix électroniques (ESL) sur le bord des tablettes.

Une étiquette = un cadre plastique blanc entourant un petit écran gris (e-ink) où
s'affiche le prix. Chaque étiquette correspond à un emplacement produit : le produit
est rangé sur la tablette juste au-dessus.

Méthode 100 % OpenCV (aucun modèle) :
  1. masque des pixels blancs peu saturés (le cadre) ;
  2. contours « anneaux » : un cadre blanc qui contient un trou (l'écran) ;
  3. filtres de forme (taille, proportions, écran plus sombre que le cadre) ;
  4. cohérence : les étiquettes vont par rangées alignées de tailles proches.
"""

from __future__ import annotations

import cv2
import numpy as np


def _ring_candidates(img: np.ndarray) -> list[dict]:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s, v = hsv[..., 1], hsv[..., 2]
    # Seuil « blanc » adaptatif à l'exposition de l'image
    v_thr = max(150, int(np.percentile(v, 80)) - 25)
    white = ((v >= v_thr) & (s <= 60)).astype(np.uint8) * 255
    k3 = np.ones((3, 3), np.uint8)
    # Le cadre se colle souvent aux reflets blancs du rail (=> ouverture pour séparer)
    # ou est incomplet sur un bord (=> fermeture pour le reboucher) : on essaie les trois
    # masques et on fusionne les détections.
    masks = [cv2.morphologyEx(white, cv2.MORPH_OPEN, k3), white,
             cv2.morphologyEx(white, cv2.MORPH_CLOSE, k3)]
    out: list[dict] = []
    for m in masks:
        for cand in _candidates_from_mask(m, gray, v, w):
            if all(_iou(cand["bbox"], o["bbox"]) < 0.4 for o in out):
                out.append(cand)
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _candidates_from_mask(white: np.ndarray, gray: np.ndarray, v: np.ndarray, w: int) -> list[dict]:
    # Le cadre est souvent incomplet (bord supérieur fin/flou) : on ne demande pas un
    # anneau fermé, seulement un bloc blanc rectangulaire au centre nettement plus sombre.
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w, max_w = 0.04 * w, 0.16 * w
    out = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if not (min_w <= bw <= max_w) or bh < 0.5 * min_w:
            continue
        ar = bw / float(bh)
        if not (0.9 <= ar <= 2.4):
            continue
        frame_mask = white[y:y + bh, x:x + bw] > 0
        white_frac = float(frame_mask.mean())
        if not (0.3 <= white_frac <= 0.85):  # trop plein = objet blanc, trop vide = bruit
            continue
        # bandes de bord : au moins 3 côtés majoritairement blancs
        t = max(2, int(round(0.15 * min(bw, bh))))
        sides = [frame_mask[:t].mean(), frame_mask[-t:].mean(),
                 frame_mask[:, :t].mean(), frame_mask[:, -t:].mean()]
        if sum(s_ > 0.55 for s_ in sides) < 3:
            continue
        # écran = rectangle intérieur, nettement plus sombre que le cadre
        mx, my = int(0.22 * bw), int(0.25 * bh)
        screen = gray[y + my:y + bh - my, x + mx:x + bw - mx]
        if screen.size == 0:
            continue
        frame_v = float(np.median(v[y:y + bh, x:x + bw][frame_mask]))
        if float(np.median(screen)) > frame_v - 25:
            continue
        out.append({"bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                    "screen": [float(x + mx), float(y + my), float(x + bw - mx), float(y + bh - my)]})
    return out


def _group_rows(labels: list[dict]) -> list[list[int]]:
    """Rangées d'étiquettes. Deux étiquettes voisines (écart horizontal < 4 largeurs) sont
    dans la même rangée si leurs centres sont à moins de 0,8 hauteur verticalement : on
    chaîne de proche en proche, ce qui suit une rangée filmée de biais (inclinée)."""
    n = len(labels)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = labels[i]["bbox"]
        ah, aw = a[3] - a[1], a[2] - a[0]
        for j in range(i + 1, n):
            b = labels[j]["bbox"]
            gap = max(b[0] - a[2], a[0] - b[2])
            dy = abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)
            if gap < 4 * max(aw, b[2] - b[0]) and dy < 0.8 * max(ah, b[3] - b[1]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    rows = sorted(groups.values(),
                  key=lambda r: np.median([(labels[i]["bbox"][1] + labels[i]["bbox"][3]) / 2 for i in r]))
    for r in rows:
        r.sort(key=lambda i: labels[i]["bbox"][0])
    return rows


def _synthetic_template(tw: int = 48, th: int = 34) -> np.ndarray:
    """Étiquette type : cadre blanc, écran gris plus sombre au centre."""
    t = np.full((th, tw), 225, np.float32)
    mx, my = int(0.16 * tw), int(0.2 * th)
    t[my:th - my, mx:tw - mx] = 130
    t[int(th * 0.45):th - my - 2, mx + 3:tw - mx - 3] = 70  # chiffres du prix
    return t


def _template_pass(img: np.ndarray, seeds: list[dict]) -> list[dict]:
    """Recherche par ressemblance : les étiquettes déjà trouvées servent de modèle
    (ou un modèle synthétique si aucune n'a été trouvée, ex. image prise de près)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    if len(seeds) >= 1:
        ws = [s["bbox"][2] - s["bbox"][0] for s in seeds]
        hs = [s["bbox"][3] - s["bbox"][1] for s in seeds]
        tw, th = int(np.median(ws)), int(np.median(hs))
        crops = []
        for s in seeds:
            x1, y1, x2, y2 = (int(round(v)) for v in s["bbox"])
            crops.append(cv2.resize(gray[y1:y2, x1:x2], (tw, th)))
        tpl = np.mean(crops, axis=0).astype(np.float32)
        # large plage d'échelles : de près, les étiquettes grossissent vers le bord (perspective)
        sizes = [(int(tw * f), int(th * f)) for f in (0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7)]
        thr = 0.6
    else:
        tpl = _synthetic_template()
        sizes = [(int(sw), int(sw / 1.4)) for sw in np.geomspace(0.04 * W, 0.14 * W, 9)]
        thr = 0.6
    found: list[tuple[float, list[float]]] = []
    for sw, sh in sizes:
        if sw < 8 or sh < 6 or sw >= W or sh >= H:
            continue
        t = cv2.resize(tpl, (sw, sh))
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        peaks = (res >= thr) & (res == cv2.dilate(res, np.ones((5, 5), np.uint8)))
        for y, x in zip(*np.nonzero(peaks)):
            found.append((float(res[y, x]), [float(x), float(y), float(x + sw), float(y + sh)]))
    found.sort(key=lambda f: -f[0])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out: list[dict] = []
    for score, b in found:
        if any(_overlap(b, o["bbox"]) > 0.2 for o in out):
            continue
        if not _looks_like_label(gray, hsv, b):
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        mx, my = 0.22 * bw, 0.25 * bh
        out.append({"bbox": b, "screen": [b[0] + mx, b[1] + my, b[2] - mx, b[3] - my],
                    "score": round(score, 3)})
    return out


def _overlap(a, b) -> float:
    """Intersection / plus petite surface (détecte les doublons décalés)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / m if m > 0 else 0.0


def _looks_like_label(gray: np.ndarray, hsv: np.ndarray, b) -> bool:
    """Cadre blanc neutre (non coloré) + écran gris neutre nettement plus sombre."""
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    g, s = gray[y1:y2, x1:x2], hsv[y1:y2, x1:x2, 1]
    bh, bw = g.shape
    if bh < 6 or bw < 8:
        return False
    mx, my = max(1, int(0.22 * bw)), max(1, int(0.25 * bh))
    tx, ty = max(1, mx // 2), max(1, my // 2)
    ring = np.zeros_like(g, bool)
    ring[:ty], ring[-ty:], ring[:, :tx], ring[:, -tx:] = True, True, True, True
    inner = np.zeros_like(g, bool)
    inner[my:bh - my, mx:bw - mx] = True
    if not inner.any():
        return False
    ring_v, ring_s = np.median(g[ring]), np.median(s[ring])
    in_v, in_s = np.median(g[inner]), np.median(s[inner])
    white_frac = float((g[ring] > ring_v - 30).mean())
    # Écran e-ink : fond GRIS (+ chiffres noirs) sur l'essentiel de la surface.
    # Un texte imprimé sur un emballage est sur fond BLANC → rejeté.
    screen_frac = float((g[inner] <= ring_v - 30).mean())
    return (ring_v >= 150 and ring_s <= 45 and in_s <= 60
            and in_v <= ring_v - 25 and white_frac >= 0.7 and screen_frac >= 0.6)


def _snap(white: np.ndarray, gray: np.ndarray, hsv: np.ndarray, b: list[float]) -> list[float]:
    """Recale le cadre sur le bloc blanc réel de l'étiquette (le modèle peut tomber à côté)."""
    H, W = gray.shape
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    wx1, wy1 = int(max(0, x1 - 0.5 * bw)), int(max(0, y1 - 0.5 * bh))
    wx2, wy2 = int(min(W, x2 + 0.5 * bw)), int(min(H, y2 + 0.5 * bh))
    win = white[wy1:wy2, wx1:wx2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
    best, best_iou = None, 0.3
    for k in range(1, n):
        sx, sy, sw, sh, area = stats[k]
        cand = [float(wx1 + sx), float(wy1 + sy), float(wx1 + sx + sw), float(wy1 + sy + sh)]
        if not (0.7 * bw <= sw <= 1.4 * bw and 0.7 * bh <= sh <= 1.4 * bh):
            continue  # collé au rail / au mur : on garde le cadre d'origine
        iou = _iou(cand, b)
        if iou > best_iou and _looks_like_label(gray, hsv, cand):
            best, best_iou = cand, iou
    return best if best is not None else b


def _drop_close_rows(rows: list[list[int]], labels: list[dict]) -> list[list[int]]:
    """Deux rangées d'étiquettes ne peuvent pas être plus proches qu'une hauteur de produit :
    si c'est le cas, la rangée la plus courte est fausse (bas de flacons, autocollants…)."""
    def cy(r):
        return float(np.median([(labels[i]["bbox"][1] + labels[i]["bbox"][3]) / 2 for i in r]))

    def hh(r):
        return float(np.median([labels[i]["bbox"][3] - labels[i]["bbox"][1] for i in r]))

    def overlap_x(a, b):
        ax1, ax2 = min(labels[i]["bbox"][0] for i in a), max(labels[i]["bbox"][2] for i in a)
        bx1, bx2 = min(labels[i]["bbox"][0] for i in b), max(labels[i]["bbox"][2] for i in b)
        return min(ax2, bx2) - max(ax1, bx1) > 0

    keep = list(rows)
    changed = True
    while changed:
        changed = False
        for a in keep:
            for b in keep:
                if a is b or not overlap_x(a, b):
                    continue
                if abs(cy(a) - cy(b)) < 3.5 * max(hh(a), hh(b)):
                    weaker = a if (len(a), cy(a)) < (len(b), cy(b)) else b
                    keep.remove(weaker)
                    changed = True
                    break
            if changed:
                break
    return keep


def filter_labels_by_items(labels: list[dict], items: list[dict], min_per_row: int = 3) -> list[dict]:
    """Retire les fausses étiquettes posées SUR un produit détecté (bas d'un tube, autocollant
    blanc et gris d'une boîte…). Une vraie étiquette est sur le rail, sous les produits."""
    def inside(l, it):
        a, b = l["bbox"], it["bbox"]
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        la = (a[2] - a[0]) * (a[3] - a[1])
        ia = (b[2] - b[0]) * (b[3] - b[1])
        return la > 0 and ix * iy / la >= 0.7 and ia >= 3 * la

    kept = [l for l in labels if not any(inside(l, it) for it in items)]
    if len(kept) == len(labels):
        return labels
    out: list[dict] = []
    for r in sorted({l["rangee"] for l in kept}):
        row = sorted([l for l in kept if l["rangee"] == r], key=lambda l: l["bbox"][0])
        if len(row) < min_per_row:
            continue
        for k, l in enumerate(row):
            out.append({**l, "index": k + 1})
    ranks = {r: i + 1 for i, r in enumerate(sorted({l["rangee"] for l in out}))}
    return [{**l, "rangee": ranks[l["rangee"]]} for l in out]


def detect_labels(img: np.ndarray, min_per_row: int = 3) -> list[dict]:
    """Retourne les étiquettes : {"bbox", "screen", "rangee", "index"} triées."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cands = [c for c in _ring_candidates(img) if _looks_like_label(gray, hsv, c["bbox"])]
    for extra in _template_pass(img, cands):
        if all(_iou(extra["bbox"], c["bbox"]) < 0.25 for c in cands):
            cands.append(extra)
    if not cands:
        return []
    hsv_s, v = hsv[..., 1], hsv[..., 2]
    v_thr = max(150, int(np.percentile(v, 80)) - 25)
    white = ((v >= v_thr) & (hsv_s <= 60)).astype(np.uint8) * 255
    for c in cands:
        nb = _snap(white, gray, hsv, c["bbox"])
        if nb is not c["bbox"]:
            bw, bh = nb[2] - nb[0], nb[3] - nb[1]
            c["bbox"] = nb
            c["screen"] = [nb[0] + 0.22 * bw, nb[1] + 0.25 * bh, nb[2] - 0.22 * bw, nb[3] - 0.25 * bh]
    # doublons créés par le recalage
    uniq: list[dict] = []
    for c in cands:
        if all(_iou(c["bbox"], u["bbox"]) < 0.5 for u in uniq):
            uniq.append(c)
    cands = uniq
    med_h = float(np.median([c["bbox"][3] - c["bbox"][1] for c in cands]))
    # tolérance large : de près, la perspective fait varier la taille des étiquettes d'une même rangée
    cands = [c for c in cands if 0.5 * med_h <= c["bbox"][3] - c["bbox"][1] <= 2.2 * med_h]
    rows = _group_rows(cands)
    # dans une rangée, la taille varie doucement (perspective) : une étiquette beaucoup plus
    # grande ou plus petite que ses voisines immédiates est un faux positif
    clean_rows = []
    for r in rows:
        hs = [cands[i]["bbox"][3] - cands[i]["bbox"][1] for i in r]
        keep = []
        for k, i in enumerate(r):
            neigh = [hs[j] for j in (k - 1, k + 1) if 0 <= j < len(r)]
            if neigh and not (0.67 <= hs[k] / float(np.median(neigh)) <= 1.5):
                continue
            keep.append(i)
        clean_rows.append(keep)
    rows = _drop_close_rows([r for r in clean_rows if len(r) >= min_per_row], cands)
    labels: list[dict] = []
    r_idx = 0
    for row in rows:
        if len(row) < min_per_row:  # étiquette isolée : souvent un faux positif
            continue
        r_idx += 1
        for k, i in enumerate(row):
            labels.append({**cands[i], "rangee": r_idx, "index": k + 1})
    return labels
