"""Carte du MUR (fond du rayon) et trous trouvés SANS étiquette de prix.

Idée : un trou, c'est un endroit où l'on voit le mur du fond là où il pourrait y avoir un
produit. On combine deux indices, tous deux locaux et gratuits :
  * la PROFONDEUR (Depth Anything V2, déjà utilisé) : le mur est nettement plus loin que la
    façade des boîtes voisines ;
  * la TEXTURE : le mur est une surface unie (gris / blanc, sans texte ni motif).
Il faut les DEUX : une boîte ratée par le détecteur est texturée et à l'avant → pas un mur,
donc pas une fausse alerte.

La « façade attendue » est interpolée entre les boîtes détectées (interpolation linéaire
entre leurs centres) : une photo prise de biais a une profondeur qui varie fortement de
gauche à droite, on ne peut donc comparer un point qu'à ses boîtes voisines.

Ensuite :
  1. les boîtes posées les unes sur les autres forment une PILE ;
  2. les piles alignées sur un même bas forment une TABLETTE ;
  3. ESPACE : entre deux piles voisines d'une tablette, on voit le mur sur une largeur d'au
     moins ~0,6 boîte → produit(s) manquant(s) ;
  4. PILE INCOMPLÈTE : au-dessus d'une pile, on voit le mur jusqu'à la tablette du dessus sur
     au moins la hauteur d'une boîte de cette pile → on pourrait y remettre une boîte.
"""

from __future__ import annotations

import cv2
import numpy as np

TEX_PLAIN = 25.0         # texture moyenne (gradient Sobel lissé) sous laquelle c'est « uni »
WALL_RECUL = 0.08        # mur : au moins 8 % plus loin que la façade des boîtes voisines
GAP_MIN_REL = 0.6        # espace entre piles ≥ 0,6 × largeur médiane des boîtes de la tablette
GAP_WALL_FRAC = 0.40     # … dans lequel on voit le mur sur ≥ 40 % de la surface
ROW_WALL_FRAC = 0.60     # au-dessus d'une pile : une ligne est « mur » si ≥ 60 % de mur
FREE_MIN_BOX = 0.50      # pile incomplète : mur visible sur ≥ 0,5 hauteur de l'objet du dessus
                         # (le détecteur encadre souvent 2 boîtes empilées comme une seule :
                         # une demi-hauteur = souvent une boîte entière ; le jeu normal sous la
                         # tablette du dessus reste bien en dessous)
FREE_MAX_BOX = 4.0       # au-delà (ou jusqu'en haut de l'image) : pas de tablette au-dessus
STACK_X_OVERLAP = 0.5    # deux boîtes de la même pile se recouvrent sur ≥ 50 % en largeur
STACK_Y_TOL = 0.35       # … et l'une est posée sur l'autre (recouvrement ≤ 35 % de la hauteur)
STACK_GAP_MAX = 0.08     # … avec au plus 8 % d'écart entre elles
STACK_W_TOL = 0.25       # … et des largeurs à 25 % près (même produit)
SHELF_NEAR_W = 1.6       # tablette : piles voisines (≤ 1,6 largeur entre centres) reliées
SHELF_LINK_W = 5.0       # … puis morceaux recollés à travers un trou de ≤ 5 largeurs
SHELF_MERGE_TOL = 0.45   # … si le bas prolongé tombe à 0,45 × hauteur de boîte près


# ------------------------------------------------------------------ carte du mur
def texture_map(img: np.ndarray) -> np.ndarray:
    """Gradient moyen local : faible = surface unie."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    k = max(7, int(0.015 * max(img.shape[:2])) | 1)
    return cv2.blur(mag, (k, k))


def _core(b, f=0.2):
    x1, y1, x2, y2 = b
    return x1 + f * (x2 - x1), y1 + f * (y2 - y1), x2 - f * (x2 - x1), y2 - f * (y2 - y1)


def front_map(depth: np.ndarray, items: list[dict]) -> np.ndarray | None:
    """Profondeur attendue de la façade, interpolée entre les boîtes détectées."""
    from scipy.interpolate import griddata

    H, W = depth.shape
    pts, vals = [], []
    for it in items:
        x1, y1, x2, y2 = _core(it["bbox"])
        xs = np.linspace(x1, x2, 3)
        ys = np.linspace(y1, y2, 3)
        for i in range(2):          # 4 quarts de la boîte : capte la pente de la façade
            for j in range(2):
                a, b, c, d = (int(round(v)) for v in (xs[i], ys[j], xs[i + 1], ys[j + 1]))
                a, c = max(0, a), min(W, max(c, a + 1))
                b, d = max(0, b), min(H, max(d, b + 1))
                patch = depth[b:d, a:c]
                if patch.size:
                    pts.append(((a + c) / 2, (b + d) / 2))
                    vals.append(float(np.median(patch)))
    if len(pts) < 4:
        return None
    pts, vals = np.array(pts), np.array(vals)
    step = 4  # grille réduite puis agrandie : rapide
    gy, gx = np.mgrid[0:H:step, 0:W:step]
    lin = griddata(pts, vals, (gx, gy), method="linear")
    near = griddata(pts, vals, (gx, gy), method="nearest")
    small = np.where(np.isnan(lin), near, lin).astype(np.float32)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def build_wall_map(img: np.ndarray, depth: np.ndarray, items: list[dict]) -> dict | None:
    """{"mur": masque booléen, "recul": recul relatif, "uni": masque, "facade": profondeur}."""
    front = front_map(depth, items)
    if front is None:
        return None
    scale = np.maximum(np.abs(front), 1e-3)
    recul = (front - depth) / scale
    plain = texture_map(img) < TEX_PLAIN
    wall = plain & (recul >= WALL_RECUL)
    H, W = depth.shape
    for it in items:  # l'intérieur d'une boîte détectée n'est jamais du mur
        x1, y1, x2, y2 = (int(round(v)) for v in _core(it["bbox"], 0.1))
        wall[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = False
    wall = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return {"mur": wall, "recul": recul, "uni": plain, "facade": front}


# ------------------------------------------------------------------ piles et tablettes
def _x_overlap(a, b) -> float:
    ov = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    m = min(a[2] - a[0], b[2] - b[0])
    return ov / m if m > 0 else 0.0


def build_stacks(items: list[dict]) -> list[dict]:
    """Regroupe les boîtes posées les unes sur les autres. Une pile : {"items", "bbox", "top"}
    (top = boîte du dessus)."""
    stacks: list[dict] = []
    for it in sorted(items, key=lambda i: -i["bbox"][3]):      # du bas vers le haut
        x1, y1, x2, y2 = it["bbox"]
        h = y2 - y1
        host = None
        for s in stacks:
            sb = s["bbox"]
            if _x_overlap(it["bbox"], sb) < STACK_X_OVERLAP:
                continue
            # posée dessus : son bas touche le haut de la pile (petit recouvrement permis, mais
            # pas d'écart : un écart = l'épaisseur d'une tablette → c'est la tablette du dessus)
            # … et même largeur que la boîte du dessous (même produit) : sur les tablettes en
            # verre très fines, la boîte de la tablette du dessus touche presque celle du dessous
            tw = s["top"]["bbox"][2] - s["top"]["bbox"][0]
            same_w = abs((x2 - x1) - tw) <= STACK_W_TOL * max(x2 - x1, tw)
            on_top = same_w and -STACK_GAP_MAX * h <= y2 - sb[1] <= STACK_Y_TOL * h
            inside = y1 >= sb[1] - 0.1 * h and y2 <= sb[3] + 0.1 * h   # doublon dans la pile
            if on_top or inside:
                host = s
                break
        if host is None:
            stacks.append({"items": [it], "bbox": list(it["bbox"]), "top": it})
            continue
        host["items"].append(it)
        sb = host["bbox"]
        host["bbox"] = [min(sb[0], x1), min(sb[1], y1), max(sb[2], x2), max(sb[3], y2)]
        if y1 < host["top"]["bbox"][1]:
            host["top"] = it
    return stacks


def _bottom_h(s: dict) -> float:
    low = max(s["items"], key=lambda i: i["bbox"][3])
    return low["bbox"][3] - low["bbox"][1]


def _cx(s: dict) -> float:
    return (s["bbox"][0] + s["bbox"][2]) / 2


def _slope(seg: list[dict]) -> float | None:
    """Pente du bas de la tablette (px de y par px de x), si le morceau est assez long."""
    if len(seg) < 3:
        return None
    xs = np.array([_cx(s) for s in seg])
    if np.ptp(xs) < 1:
        return None
    return float(np.polyfit(xs, [s["bbox"][3] for s in seg], 1)[0])


def build_shelves(stacks: list[dict]) -> list[list[dict]]:
    """Piles dont les bas sont alignés → une tablette (de haut en bas, piles de gauche à droite).

    1. on relie les piles VOISINES (≤ 1,6 largeur entre centres) dont les bas sont alignés,
       de proche en proche : la tablette peut être penchée (photo de biais) ;
    2. on recolle deux morceaux séparés par un trou (≤ 5 largeurs) seulement si l'un est dans
       le PROLONGEMENT de l'autre (pente de la tablette prise en compte). Relier directement des
       piles lointaines mélangerait les tablettes sur une photo penchée."""
    n = len(stacks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cx = [_cx(s) for s in stacks]
    w = [s["bbox"][2] - s["bbox"][0] for s in stacks]
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(cx[i] - cx[j])
            if dx > SHELF_NEAR_W * max(w[i], w[j]):
                continue
            tol = 0.3 * min(_bottom_h(stacks[i]), _bottom_h(stacks[j])) + 0.25 * dx
            if abs(stacks[i]["bbox"][3] - stacks[j]["bbox"][3]) <= tol:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, s in enumerate(stacks):
        groups.setdefault(find(i), []).append(s)
    segs = [sorted(g, key=_cx) for g in groups.values()]

    slopes = [sl for sl in (_slope(g) for g in segs) if sl is not None]
    default = float(np.median(slopes)) if slopes else 0.0
    merged = True
    while merged:                   # recoller les morceaux séparés par un trou
        merged = False
        for a in segs:
            for b in segs:
                if a is b:
                    continue
                ra, lb = a[-1], b[0]
                gap = lb["bbox"][0] - ra["bbox"][2]
                wmax = max(ra["bbox"][2] - ra["bbox"][0], lb["bbox"][2] - lb["bbox"][0])
                if gap < -0.3 * wmax or gap > SHELF_LINK_W * wmax:
                    continue
                sl = _slope(a) if _slope(a) is not None else (_slope(b) if _slope(b) is not None else default)
                pred = ra["bbox"][3] + sl * (_cx(lb) - _cx(ra))
                if abs(pred - lb["bbox"][3]) <= SHELF_MERGE_TOL * min(_bottom_h(ra), _bottom_h(lb)):
                    a.extend(b)
                    a.sort(key=_cx)
                    segs.remove(b)
                    merged = True
                    break
            if merged:
                break
    shelves = [sorted(g, key=lambda s: s["bbox"][0]) for g in segs]
    shelves.sort(key=lambda g: float(np.median([s["bbox"][3] for s in g])))
    return shelves


# ------------------------------------------------------------------ trous
def _frac(mask: np.ndarray, box) -> float:
    H, W = mask.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())


def _hole(zone, etat, shelf_no, index, manquants, mesures, produits=()) -> dict:
    zx1, zy1, zx2, zy2 = (round(float(v), 1) for v in zone)
    return {"bbox": [zx1, zy2, zx2, zy2 + 4.0],   # pas d'étiquette : repère = bas de la zone
            "zone": [zx1, zy1, zx2, zy2], "etat": etat, "rangee": shelf_no, "index": index,
            "manquants": int(manquants), "produits": list(produits), "mesures": mesures,
            "screen": None, "source": "mur"}


def find_gaps_between_stacks(wall: np.ndarray, shelf: list[dict], shelf_no: int) -> list[dict]:
    if len(shelf) < 2:
        return []
    med_w = float(np.median([s["top"]["bbox"][2] - s["top"]["bbox"][0] for s in shelf]))
    med_h = float(np.median([s["top"]["bbox"][3] - s["top"]["bbox"][1] for s in shelf]))
    out = []
    reach = shelf[0]
    for k, nxt in enumerate(shelf[1:], start=1):
        x_left, x_right = reach["bbox"][2], nxt["bbox"][0]
        gap = x_right - x_left
        if gap >= GAP_MIN_REL * med_w:
            y_bot = (reach["bbox"][3] + nxt["bbox"][3]) / 2
            y_top = min((reach["bbox"][1] + nxt["bbox"][1]) / 2, y_bot - med_h)
            h = y_bot - y_top
            core = [x_left + 0.1 * gap, y_top + 0.15 * h, x_right - 0.1 * gap, y_bot - 0.1 * h]
            f = _frac(wall, core)
            if f >= GAP_WALL_FRAC:
                n = max(1, int(round(gap / med_w)))
                out.append(_hole([x_left + 1, y_top, x_right - 1, y_bot], "espace", shelf_no, k, n,
                                 {"mur": round(f, 2), "places_libres": n,
                                  "entre_produits": [reach["top"]["id"], nxt["top"]["id"]]}))
        if nxt["bbox"][2] > reach["bbox"][2]:
            reach = nxt
    return out


def _shelf_above(items: list[dict], stack: dict, xa: int, xb: int, end_y: int, box_h: float) -> bool:
    """Des produits sont-ils posés sur une tablette juste au-dessus du mur vu ? Sinon c'est le
    haut du meuble (plafond, enseigne, fond du magasin) : pas une place libre."""
    own = {id(it) for it in stack["items"]}
    pad = 0.5 * (xb - xa)
    lo, hi = end_y - max(0.8 * box_h, 25.0), end_y + 0.3 * box_h
    for it in items:
        if id(it) in own:
            continue
        x1, _, x2, y2 = it["bbox"]
        if x2 >= xa - pad and x1 <= xb + pad and lo <= y2 <= hi and y2 <= stack["bbox"][1]:
            return True
    return False


def find_low_stacks(wall: np.ndarray, shelf: list[dict], shelf_no: int,
                    items: list[dict] | None = None) -> list[dict]:
    """Au-dessus de chaque pile : hauteur de mur visible avant d'atteindre la tablette du dessus."""
    H, W = wall.shape
    items = items if items is not None else [i for s in shelf for i in s["items"]]
    out = []
    for k, s in enumerate(shelf, start=1):
        tb = s["top"]["bbox"]
        box_h = tb[3] - tb[1]
        sx1, sx2 = s["bbox"][0], s["bbox"][2]
        xa, xb = int(sx1 + 0.15 * (sx2 - sx1)), int(sx2 - 0.15 * (sx2 - sx1))
        xa, xb = max(0, xa), min(W, xb)
        if xb - xa < 3 or box_h < 6:
            continue
        top = int(s["bbox"][1])
        skip = max(3, int(0.35 * box_h))     # dessus de la boîte (vu en plongée), ombre…
        run, misses, started, end_y, blocked = 0, 0, False, top, False
        for y in range(top - 1, max(-1, top - 1 - int(FREE_MAX_BOX * box_h)), -1):
            if wall[y, xa:xb].mean() >= ROW_WALL_FRAC:
                started, misses, run, end_y = True, 0, top - y, y
                continue
            misses += 1
            if (started and misses >= 3) or (not started and top - y > skip):
                blocked = True        # on a atteint autre chose que le mur (tablette du dessus)
                break
        if not blocked or run < FREE_MIN_BOX * box_h:
            continue                  # rien vu, ou mur jusqu'en haut : pas de tablette au-dessus
        if not _shelf_above(items, s, xa, xb, end_y, box_h):
            continue
        n = max(1, int(round(run / box_h)))
        out.append(_hole([xa, end_y, xb, top], "incomplet", shelf_no, k, n,
                         {"mur_au_dessus_px": run, "hauteur_boite_px": round(box_h, 1),
                          "places_libres": n, "pile": [it["id"] for it in s["items"]]},
                         produits=[it["id"] for it in s["items"]]))
    return out


def find_wall_holes(img: np.ndarray, depth: np.ndarray, items: list[dict]) -> tuple[list[dict], dict | None]:
    """Trous trouvés grâce au mur, sans étiquette. Renvoie (trous, carte du mur)."""
    if depth is None or len(items) < 2:
        return [], None
    wm = build_wall_map(img, depth, items)
    if wm is None:
        return [], None
    holes = []
    for no, shelf in enumerate(build_shelves(build_stacks(items)), start=1):
        for it in (i for s in shelf for i in s["items"]):
            it["tablette"] = no
        holes += find_gaps_between_stacks(wm["mur"], shelf, no)
        holes += find_low_stacks(wm["mur"], shelf, no, items)
    return holes, wm


WALL_TINT = (80, 200, 80)   # vert (BGR)


def tint_wall(img: np.ndarray, wall: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Superpose la carte du mur (vert translucide) pour la preuve."""
    out = img.copy()
    color = np.zeros_like(img)
    color[:] = WALL_TINT
    blend = cv2.addWeighted(img, 1 - alpha, color, alpha, 0)
    out[wall] = blend[wall]
    return out
