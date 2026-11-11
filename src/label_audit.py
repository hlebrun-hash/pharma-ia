"""Audit « par étiquettes » : trouver les trous d'un rayon SANS photo de référence.

Principe (une étiquette de prix = un emplacement produit) :
  1. on repère les étiquettes de prix électroniques sur le bord des tablettes ;
  2. on regarde la zone juste au-dessus de chaque étiquette :
       - YOLO y voit-il un produit ?  - y a-t-il des contours / des couleurs d'emballage ?
  3. si rien de tout ça → emplacement VIDE → à réassortir ;
  4. sinon, une carte de profondeur regarde entre l'avant du rayon et le mur :
       - ESPACE : on voit le fond sur une bonne partie de la largeur → un produit manque ;
       - PILE INCOMPLÈTE : des produits sont à l'avant mais on voit le mur au-dessus ;
       - RETRAIT : la façade est loin derrière l'étiquette → les boîtes de devant manquent.

Sans étiquette (ou en plus) : une CARTE DU MUR (profondeur + surface unie, voir
wall_map.py) montre où l'on voit le fond du rayon entre deux piles de produits ou au-dessus
d'une pile → produit manquant / pile incomplète.

Pour une vidéo, on analyse une image nette toutes les `interval` secondes ; un même trou
vu sur plusieurs images consécutives n'est compté qu'une fois.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from src import OUTPUTS_DIR, ROOT_DIR
from src.shelf_detector import detect_boxes
from src.label_detector import detect_labels, filter_labels_by_items
from src.shelf_items import assign_items, draw_proof, find_item_gaps, new_gaps_only
from src.slot_checker import ANOMALY_COLORS, check_frame, draw
from src.video_engine import extract_sharp_frames, is_video, sharpness_score
from src.wall_map import find_wall_holes

TYPES = {  # état de l'emplacement → (type_anomalie du manifeste, libellé, gravité)
    "vide": ("EMPLACEMENT_VIDE", "Emplacement vide", 3),
    "espace": ("ESPACE_PRODUIT_MANQUANT", "Espace : produit manquant", 2),
    "incomplet": ("PILE_INCOMPLETE", "Pile incomplète : on voit le mur au-dessus des produits", 2),
    "retrait": ("PRODUIT_EN_RETRAIT", "Produits de devant manquants (façade en retrait)", 1),
}
SAME_SLOT_NCC = 0.5      # ressemblance minimale pour dire « c'est le même trou » (secours)
SAME_SLOT_MAX_GAP = 4    # … entre deux images distantes d'au plus 4 positions
BLUR_RATIO = 0.55        # image ignorée si sa netteté < 55 % de la netteté médiane de la vidéo
WORK_MAX_W = 1920        # analyse sur une image d'au plus 1920 px de large ; le texte des
                         # étiquettes est lu dans l'image d'origine (4K) pour rester lisible
FRAME_MAX_W = 3840       # images extraites de la vidéo en pleine résolution (jusqu'à la 4K)


def _rel(p: str | Path | None) -> str | None:
    if not p:
        return None
    try:
        return Path(p).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(p)


def _context_box(slot: dict, shape) -> list[int]:
    """Zone + étiquette + un peu de contexte autour (pour la preuve et le dédoublonnage)."""
    H, W = shape[:2]
    lx1, ly1, lx2, ly2 = slot["bbox"]
    zx1, zy1, zx2, zy2 = slot["zone"]
    lw, lh = lx2 - lx1, ly2 - ly1
    x1, x2 = min(lx1, zx1) - 0.8 * lw, max(lx2, zx2) + 0.8 * lw
    y1, y2 = zy1 - 0.3 * lh, ly2 + 0.4 * lh
    if slot.get("source") == "mur":  # pas d'étiquette : on montre aussi les produits autour
        zh = zy2 - zy1
        x1, x2 = zx1 - 1.2 * (zx2 - zx1), zx2 + 1.2 * (zx2 - zx1)
        y1, y2 = zy1 - 0.5 * zh, zy2 + 1.2 * zh
    return [int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))]


def _signature(img: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    n = float(np.linalg.norm(g))
    return g / n if n > 0 else g


class _FrameMatcher:
    """Retrouve un même endroit du rayon sur deux images de la vidéo : points d'intérêt ORB
    appariés + homographie RANSAC. Bien plus fiable que la ressemblance des pixels quand la
    caméra bouge (le même trou vu de face puis de biais ne se ressemble pas pixel à pixel)."""

    def __init__(self, max_side: int = 960):
        self.max_side = max_side
        self.orb = cv2.ORB_create(3000)
        self.feats: dict[int, tuple] = {}
        self.cache: dict[tuple[int, int], np.ndarray | None] = {}

    def add(self, fi: int, img: np.ndarray) -> None:
        sc = min(1.0, self.max_side / max(img.shape[:2]))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if sc < 1.0:
            g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        kp, des = self.orb.detectAndCompute(g, None)
        self.feats[fi] = (kp, des, sc)

    def homography(self, a: int, b: int) -> np.ndarray | None:
        """Matrice qui envoie un point de l'image a vers l'image b (repère d'analyse)."""
        if (a, b) in self.cache:
            return self.cache[(a, b)]
        H = None
        fa, fb = self.feats.get(a), self.feats.get(b)
        if fa and fb and fa[1] is not None and fb[1] is not None and len(fa[0]) >= 30 and len(fb[0]) >= 30:
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(fa[1], fb[1], k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) >= 25:
                src = np.float32([fa[0][m.queryIdx].pt for m in good])
                dst = np.float32([fb[0][m.trainIdx].pt for m in good])
                Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                if Hs is not None and mask is not None:
                    inl = int(mask.sum())
                    det = float(np.linalg.det(Hs[:2, :2]))
                    if inl >= 20 and inl >= 0.2 * len(good) and 0.3 < det < 3.0 \
                            and abs(Hs[2, 0]) < 2e-3 and abs(Hs[2, 1]) < 2e-3:
                        Sa = np.diag([fa[2], fa[2], 1.0])
                        Sb_inv = np.diag([1 / fb[2], 1 / fb[2], 1.0])
                        H = Sb_inv @ Hs @ Sa
        self.cache[(a, b)] = H
        return H

    def same_place(self, a: int, zone_a, b: int, zone_b) -> bool | None:
        """True / False si l'homographie a pu être calculée, None sinon."""
        H = self.homography(a, b)
        if H is None:
            return None
        x1, y1, x2, y2 = zone_a
        pts = cv2.perspectiveTransform(
            np.float32([[[x1, y1]], [[x2, y2]], [[(x1 + x2) / 2, (y1 + y2) / 2]]]), H)
        (px1, _), (px2, _), (cx, cy) = pts[0][0], pts[1][0], pts[2][0]
        bx1, by1, bx2, by2 = zone_b
        mx, my = 0.25 * (bx2 - bx1), 0.25 * (by2 - by1)
        inside = bx1 - mx <= cx <= bx2 + mx and by1 - my <= cy <= by2 + my
        wa, wb = abs(px2 - px1), bx2 - bx1
        return bool(inside and wb > 0 and 0.4 <= wa / wb <= 2.5)


def _label_name(slot: dict) -> str:
    ident = slot.get("identification") or {}
    return (ident.get("etiquette") or {}).get("nom", "") or ""


def _same_hole(g: dict, s: dict, fi: int, sig: np.ndarray, matcher: _FrameMatcher) -> bool:
    """Le trou `s` (image fi) est-il le même que celui du groupe `g` ?"""
    if g["last_frame"] == fi:   # deux trous d'une même image sont distincts
        return False
    na, nb = _label_name(g["slot"]), _label_name(s)
    if len(na) >= 6 and len(nb) >= 6:   # les deux étiquettes ont été lues : elles tranchent
        from rapidfuzz import fuzz
        r = fuzz.ratio(na, nb)
        if r >= 85:
            return True
        if r <= 50:
            return False
    if fi - g["last_frame"] > SAME_SLOT_MAX_GAP:
        return False
    geo = matcher.same_place(fi, s["zone"], g["last_frame"], g["last_zone"])
    if geo is not None:
        return geo
    return float((g["sig"] * sig).sum()) >= SAME_SLOT_NCC


def _keep_worst(g: dict, s: dict) -> None:
    """Le groupe garde le constat le plus grave, et un nom de produit lu sur n'importe quelle vue."""
    if TYPES[s["etat"]][2] > TYPES[g["slot"]["etat"]][2]:
        keep_ident = g["slot"].get("identification")
        g["slot"] = s
        if keep_ident and keep_ident.get("nom") and not (s.get("identification") or {}).get("nom"):
            s["identification"] = keep_ident
    elif (s.get("identification") or {}).get("nom") and not \
            (g["slot"].get("identification") or {}).get("nom"):
        g["slot"]["identification"] = s["identification"]  # nom lu sur une autre vue


def _merge_groups(main: dict, other: dict) -> None:
    """Deux groupes reconnus comme le même trou (via une vue qui ressemble aux deux)."""
    main["vues"] = sorted(main["vues"] + other["vues"], key=lambda v: str(v["image"]))
    if other["last_frame"] > main["last_frame"]:
        main["last_frame"], main["sig"], main["last_zone"] = \
            other["last_frame"], other["sig"], other["last_zone"]
    _keep_worst(main, other["slot"])


def _frame_time(path: str, fps: float | None) -> float | None:
    m = re.search(r"_f(\d+)\.jpg$", str(path))
    if not m or not fps:
        return None
    return int(m.group(1)) / fps


def _fmt_time(t: float | None) -> str:
    if t is None:
        return ""
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _montage(images: list[np.ndarray], cols: int = 3, height: int = 520) -> np.ndarray:
    tiles = [cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height)) for im in images]
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 4, 4, 4, 4 + w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols and len(tiles) > cols:
            row.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(row))
    return np.vstack(rows)


ITEMS_CACHE_VERSION = 2


def _cache_key(fpath: str, version: int, work_w: int | None) -> dict:
    # clé = contenu de l'image (la vidéo est redécoupée à chaque analyse : même image,
    # nouveau fichier) → pas de recalcul inutile ; + largeur d'analyse si l'image est réduite
    key = {"v": version, "md5": hashlib.md5(Path(fpath).read_bytes()).hexdigest()}
    if work_w:
        key["w"] = work_w
    return key


def _cached_items(fpath: str, img, yolo_boxes, slots, item_fn, work_w: int | None = None) -> list[dict]:
    """La détection de chaque produit est lente : résultat mis en cache à côté de l'image."""
    cache = Path(str(fpath) + ".produits.json")
    key = _cache_key(fpath, ITEMS_CACHE_VERSION, work_w)
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                return data["items"]
        except Exception:
            pass
    items = item_fn(img, yolo_boxes, slots)
    cache.write_text(json.dumps({"key": key, "items": items}), encoding="utf-8")
    return items


ROW_ALL_EMPTY_MIN = 3   # rangée d'au moins 3 « étiquettes » toutes vides → pas une vraie tablette


def _drop_all_empty_rows(slots: list[dict]) -> tuple[list[dict], int]:
    """Écarte les rangées où TOUS les emplacements sont vides. Sur « teste V2 », les rainures
    horizontales d'un mur blanc ressemblaient à des étiquettes : rien au-dessus → que des
    « vides ». Une tablette réelle entièrement vide est très rare, et se voit à l'œil nu."""
    rows: dict[int, list[dict]] = {}
    for s in slots:
        rows.setdefault(s["rangee"], []).append(s)
    bad = {r for r, ss in rows.items()
           if len(ss) >= ROW_ALL_EMPTY_MIN and all(s["etat"] == "vide" for s in ss)}
    if not bad:
        return slots, 0
    kept = [s for s in slots if s["rangee"] not in bad]
    order = sorted({s["rangee"] for s in kept})
    renum = {r: i + 1 for i, r in enumerate(order)}
    for s in kept:
        s["rangee"] = renum[s["rangee"]]
    return kept, len(bad)


TEXT_CACHE_VERSION = 2


def _cached_text(fpath: str, img, items: list[dict], work_w: int | None = None) -> list[dict]:
    """Lecture de tout le texte de l'image (OCR, une passe), mise en cache à côté de l'image,
    puis attribution de chaque ligne au produit qui la contient. Renvoie les lignes lues."""
    from src.text_reader import read_frame, read_items

    cache = Path(str(fpath) + ".texte.json")
    key = _cache_key(fpath, TEXT_CACHE_VERSION, work_w)
    lines = None
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                lines = data["lignes"]
        except Exception:
            pass
    if lines is None:
        lines = read_frame(img)
        cache.write_text(json.dumps({"key": key, "lignes": lines}, ensure_ascii=False), encoding="utf-8")
    read_items(img, items, lines)
    return lines


def _identify_slot(img, slot: dict, items: list[dict], lines: list[dict] | None = None,
                   img_full=None, scale: float = 1.0) -> dict:
    """Nom du produit à réassortir. Ordre de confiance :
    1. l'étiquette de prix électronique, lue en pleine résolution, reconnue dans la base ;
    2. les produits restants sur l'emplacement (même pile → même produit), reconnus dans la base ;
    3. le texte brut de l'étiquette (« SEBIUM KERATO+ BODY SPRAY ») puis celui des produits ;
    4. à défaut, on donne les produits voisins pour aider à le retrouver."""
    from collections import Counter

    from src.text_reader import identify, lines_in, read_label_full

    by_id = {it["id"]: it for it in items}
    on_slot = [by_id[i] for i in slot.get("produits", []) if i in by_id]
    names = [it.get("nom_lu") for it in on_slot if it.get("nom_lu")]
    texts_slot = [n for n, _ in Counter(names).most_common()]
    for it in on_slot:
        texts_slot += it.get("texte", [])
    if lines:  # texte lu dans la zone de l'emplacement même sans boîte détectée (pile en retrait…)
        texts_slot += [l["texte"] for l in sorted(lines_in(lines, slot["zone"], pad=0.0),
                                                   key=lambda l: -l["hauteur"])]
    label = {"nom": "", "prix": None, "lignes": [], "lisible": False}
    if slot.get("source") != "mur":   # trouvé sans étiquette : rien à lire en dessous
        label = read_label_full(img_full if img_full is not None else img, slot["bbox"], scale)
    label_txt = [label["nom"]] if label["nom"] else []
    near_ids = list(slot.get("mesures", {}).get("entre_produits", []))
    if not near_ids:  # produits les plus proches à gauche et à droite sur la même tablette
        zx1, _, zx2, _ = slot["zone"]
        key = "tablette" if slot.get("source") == "mur" else "rangee"  # tablette repérée par les produits
        row = [it for it in items if it.get(key) == slot.get("rangee")]
        left = [it for it in row if it["bbox"][2] <= zx1 + 5]
        right = [it for it in row if it["bbox"][0] >= zx2 - 5]
        if left:
            near_ids.append(max(left, key=lambda it: it["bbox"][2])["id"])
        if right:
            near_ids.append(min(right, key=lambda it: it["bbox"][0])["id"])
    neighbours = list(dict.fromkeys(by_id[i].get("nom_lu", "") for i in near_ids if i in by_id))
    lab = identify(label_txt) if label_txt else {"nom": "", "source": ""}
    prod = identify(texts_slot) if texts_slot else {"nom": "", "source": ""}
    if lab.get("source") == "base":
        ident, via = lab, "etiquette"
    elif prod.get("source") == "base":
        ident, via = prod, "produits restants"
    elif lab.get("nom") and len(lab["nom"]) >= 5:
        ident, via = lab, "etiquette"
    elif prod.get("nom"):
        ident, via = prod, "produits restants"
    else:
        ident, via = {"nom": "", "source": ""}, ""
    return {"nom": ident.get("nom", ""), "source": ident.get("source", ""), "via": via,
            "base": ident.get("base"), "etiquette": label, "prix": label.get("prix"),
            "texte_etiquette": label.get("lignes", []),
            "texte_produits": texts_slot[:8], "voisins": [n for n in neighbours if n]}


def run_label_audit(input_path: str, interval: float = 1.5, detector: str = "auto",
                    weights: str | None = None, conf: float = 0.25, use_depth: bool = True,
                    detect_all_items: bool = False, read_text: bool = True,
                    out_dir: str | Path = OUTPUTS_DIR,
                    manifest_name: str = "reassort_manifest.json", progress=None) -> dict:
    """Analyse une photo ou une vidéo de rayon et liste les emplacements à réassortir.

    detect_all_items : détecte et numérote CHAQUE produit visible (Grounding DINO + YOLO),
        sert de preuve et permet de trouver les espaces entre deux produits et les trous sans
        étiquette (carte du mur). TRÈS lent sur un PC sans carte graphique (plusieurs minutes
        par image) : désactivé par défaut.
    progress : fonction optionnelle progress(fraction 0..1, message) pour l'interface.
    """
    def _progress(frac: float, msg: str) -> None:
        if progress is not None:
            try:
                progress(min(1.0, max(0.0, frac)), msg)
            except Exception:
                pass

    t0 = time.time()
    out_dir = Path(out_dir)
    proofs_dir = out_dir / "preuves"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("trou_*.jpg", "detection_*.jpg"):
        for old in proofs_dir.glob(pattern):
            old.unlink()
    warnings: list[str] = []

    fps = None
    if is_video(input_path):
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()
        _progress(0.0, "Extraction des images nettes de la vidéo…")
        frames = extract_sharp_frames(input_path, interval_sec=interval, max_width=FRAME_MAX_W)
        if not frames:
            raise ValueError("Aucune image exploitable extraite de la vidéo.")
    elif Path(input_path).is_dir():  # dossier d'images (photos, ou images déjà extraites)
        frames = sorted(str(p) for p in Path(input_path).iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not frames:
            raise ValueError(f"Aucune image dans le dossier {input_path}")
    else:
        frames = [str(input_path)]

    if len(frames) >= 3:  # vidéo : on écarte les images bougées (fausses étiquettes)
        sharp = {f: sharpness_score(cv2.imread(f)) for f in frames}
        med = float(np.median(list(sharp.values())))
        blurry = [f for f in frames if sharp[f] < BLUR_RATIO * med]
        if blurry:
            frames = [f for f in frames if f not in blurry]
            warnings.append(f"{len(blurry)} image(s) trop floue(s) ignorée(s) : filmez plus lentement.")

    use_yolo = detector != "classical"
    depth_fn = None
    if use_depth:
        try:
            from src.depth_estimator import estimate_depth
            depth_fn = estimate_depth
        except Exception as e:  # transformers absent, modèle non téléchargeable…
            warnings.append(f"Profondeur indisponible ({e}) : seuls les emplacements vides sont cherchés.")
    item_fn = None
    if detect_all_items:
        try:
            from src.item_detector import detect_items
            item_fn = detect_items
        except Exception as e:
            warnings.append(f"Détection de chaque produit indisponible ({e}).")
    groups: list[dict] = []      # un groupe = un trou (vu sur 1 ou plusieurs images)
    matcher = _FrameMatcher()
    detection_proofs: list[str] = []
    n_items = 0
    frame_reports, annotated_with_holes = [], []
    n_labels = 0
    n_rows_dropped = 0
    n_wall_holes = 0
    for fi, fpath in enumerate(frames):
        _progress(0.05 + 0.9 * fi / max(1, len(frames)),
                  f"Image {fi + 1}/{len(frames)} : étiquettes, produits, profondeur…")
        img_full = cv2.imread(fpath)
        if img_full is None:
            warnings.append(f"Image illisible : {fpath}")
            continue
        scale = min(1.0, WORK_MAX_W / img_full.shape[1])
        img = img_full if scale >= 1.0 else cv2.resize(img_full, None, fx=scale, fy=scale,
                                                       interpolation=cv2.INTER_AREA)
        work_w = img.shape[1] if scale < 1.0 else None
        matcher.add(fi, img)
        yolo_boxes = []
        if use_yolo:
            boxes, used = detect_boxes(img, detector, weights=weights, conf=conf)
            yolo_boxes = [b[:5] for b in boxes] if used.startswith("yolo") else []
        depth = None
        if depth_fn is not None:
            try:
                depth = depth_fn(img)
            except Exception as e:
                warnings.append(f"Profondeur indisponible ({e}) : seuls les emplacements vides sont cherchés.")
                depth_fn = None
        labels = detect_labels(img)
        items: list[dict] = []
        if item_fn is not None:
            try:
                items = _cached_items(fpath, img, yolo_boxes, labels, item_fn, work_w)
            except Exception as e:
                warnings.append(f"Détection de chaque produit indisponible ({e}).")
                item_fn = None
        text_lines: list[dict] = []
        if read_text and items:  # texte de chaque produit détecté (les étiquettes sont lues à part)
            try:
                text_lines = _cached_text(fpath, img, items, work_w)
            except Exception as e:
                warnings.append(f"Lecture du texte indisponible ({e}).")
                read_text = False
        if items:  # une « étiquette » posée sur un produit détecté n'en est pas une
            labels = filter_labels_by_items(labels, items)
        slots = check_frame(img, yolo_boxes, depth, labels)
        slots, n_drop = _drop_all_empty_rows(slots)
        n_rows_dropped += n_drop
        n_labels += len(slots)
        holes = [s for s in slots if s["etat"] in TYPES]
        if items:
            per_row = assign_items(items, slots)
            holes += new_gaps_only(find_item_gaps(img, per_row, slots, depth), holes)
        wall = None
        if items and depth is not None:  # trous vus grâce au mur, avec ou sans étiquette
            try:
                wall_holes, wm = find_wall_holes(img, depth, items)
                wall = wm["mur"] if wm else None
                new = new_gaps_only(wall_holes, holes)
                holes += new
                n_wall_holes += len(new)
            except Exception as e:
                warnings.append(f"Carte du mur indisponible ({e}).")
        n_items += len(items)
        if read_text:
            for s in holes:
                try:
                    s["identification"] = _identify_slot(img, s, items, text_lines, img_full, scale)
                except Exception as e:
                    warnings.append(f"Lecture du texte indisponible ({e}).")
                    read_text = False
                    break
        if item_fn is not None or items:
            proof_det = proofs_dir / f"detection_{fi + 1:02d}.jpg"
            cv2.imwrite(str(proof_det), draw_proof(img, items, slots, holes, wall))
            detection_proofs.append(_rel(proof_det))
        frame_reports.append({"image": _rel(fpath), "temps_s": _frame_time(fpath, fps),
                              "nb_etiquettes": len(slots), "nb_produits": len(items),
                              "nb_anomalies": len(holes), "emplacements": slots,
                              "produits": items,
                              "preuve_detection": detection_proofs[-1] if items else None})
        if holes:
            annotated_with_holes.append(draw(img, slots + [h for h in holes if h.get("source") in ("produits", "mur")]))
        for s in holes:
            box = _context_box(s, img.shape)
            sig = _signature(img, box)
            matches = [g for g in groups if _same_hole(g, s, fi, sig, matcher)]
            if matches:
                match = matches[0]
                for other in matches[1:]:   # cette vue relie deux groupes : c'est un seul trou…
                    seen = {v["image"] for v in match["vues"]}
                    if seen.isdisjoint(v["image"] for v in other["vues"]):  # …sauf s'ils sont
                        _merge_groups(match, other)                         # sur une même image
                        groups = [g for g in groups if g is not other]
                match["vues"].append({"image": _rel(fpath), "temps_s": _frame_time(fpath, fps),
                                      "etat": s["etat"], "zone": s["zone"]})
                match["last_frame"], match["sig"], match["last_zone"] = fi, sig, s["zone"]
                _keep_worst(match, s)
                continue
            proof = img[box[1]:box[3], box[0]:box[2]].copy()
            zx1, zy1, zx2, zy2 = (int(v) for v in s["zone"])
            lx1, ly1, lx2, ly2 = (int(v) for v in s["bbox"])
            color = ANOMALY_COLORS[s["etat"]]
            cv2.rectangle(proof, (zx1 - box[0], zy1 - box[1]), (zx2 - box[0], zy2 - box[1]), color, 2)
            if s.get("source") != "mur":  # pas d'étiquette à montrer pour un trou vu grâce au mur
                cv2.rectangle(proof, (lx1 - box[0], ly1 - box[1]), (lx2 - box[0], ly2 - box[1]),
                              (255, 0, 255), 2)
            proof = cv2.resize(proof, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            groups.append({"sig": sig, "last_frame": fi, "last_zone": s["zone"], "slot": s,
                           "proof_img": proof,
                           "vues": [{"image": _rel(fpath), "temps_s": _frame_time(fpath, fps),
                                     "etat": s["etat"], "zone": s["zone"]}]})

    if n_labels == 0:
        if n_items and depth_fn is not None:
            warnings.append("Aucune étiquette de prix électronique détectée : les trous ont été "
                            "cherchés uniquement grâce au mur visible entre et au-dessus des produits.")
        else:
            warnings.append("Aucune étiquette de prix détectée : filmez le rayon de face, à "
                            "environ 1 m, en gardant les étiquettes bien visibles.")

    manifest, holes_details = [], []
    det_by_image = {fr["image"]: fr.get("preuve_detection") for fr in frame_reports}
    groups.sort(key=lambda g: -TYPES[g["slot"]["etat"]][2])  # vides d'abord
    for k, g in enumerate(groups, start=1):
        proof_path = proofs_dir / f"trou_{k:02d}.jpg"
        cv2.imwrite(str(proof_path), g["proof_img"])
        first = g["vues"][0]
        when = f"à {_fmt_time(first['temps_s'])} dans la vidéo, " if first["temps_s"] is not None else ""
        s = g["slot"]
        if s.get("source") == "mur":
            where = (f"{s['index']}ᵉ pile en partant de la gauche" if s["etat"] == "incomplet"
                     else f"après la {s['index']}ᵉ pile en partant de la gauche")
            emplacement = f"{when}tablette {s['rangee']} (de haut en bas), {where}"
        else:
            emplacement = (f"{when}rangée d'étiquettes {s['rangee']} (de haut en bas), "
                           f"{s['index']}ᵉ étiquette en partant de la gauche")
        between = s.get("mesures", {}).get("entre_produits")
        if between:
            emplacement += f", entre les produits n°{between[0]} et n°{between[1]}"
        ident = s.get("identification") or {}
        if ident.get("source") == "base":
            produit = ident["nom"]
        elif ident.get("nom"):
            produit = f"{ident['nom']} (texte lu, à vérifier)"
        elif ident.get("voisins"):
            produit = f"Produit à identifier (à côté de : {', '.join(ident['voisins'])})"
        elif s.get("source") == "mur" and s["etat"] == "incomplet" and s.get("produits"):
            produit = (f"Même produit que la boîte n°{s['produits'][0]} juste en dessous "
                       f"(nom non lu, voir la photo)")
        elif s.get("source") == "mur":
            produit = "Produit à identifier (voir la photo)"
        else:
            produit = "Produit à identifier (voir l'étiquette sur la photo)"
        manifest.append({
            "id": k,
            "produit": produit,
            "emplacement": emplacement[0].upper() + emplacement[1:],
            "type_anomalie": TYPES[s["etat"]][0],
            "quantite_presente": 0,
            "capacite_totale": s.get("manquants", 1),
            "quantite_a_reassortir": s.get("manquants", 1),
            "image_preuve": _rel(proof_path),
        })
        holes_details.append({"id": k, "etat": s["etat"], "libelle": TYPES[s["etat"]][1],
                              "vu_sur_images": g["vues"], "nb_vues": len(g["vues"]),
                              "certitude": "forte" if len(g["vues"]) >= 2 else "moyenne",
                              "etiquette_bbox": s["bbox"], "zone": s["zone"], "mesures": s["mesures"],
                              "image_preuve": _rel(proof_path),
                              "preuve_detection": det_by_image.get(first["image"]),
                              "identification": ident or None,
                              "source": s.get("source", "etiquette")})

    from src.pipeline import MANIFEST_SCHEMA  # import tardif (évite l'import circulaire)
    MANIFEST_SCHEMA.validate_python(manifest)
    manifest_path = out_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if annotated_with_holes:
        overview = _montage(annotated_with_holes[:9])
    else:
        first_img = cv2.imread(frames[0])
        overview = draw(first_img, frame_reports[0]["emplacements"]) if frame_reports else first_img
    annotated_path = out_dir / "audit_annote.jpg"
    cv2.imwrite(str(annotated_path), overview)

    summary = {
        "mode": "etiquettes",
        "capture": _rel(input_path),
        "frames_analysees": [_rel(f) for f in frames],
        "detecteur": " + ".join(["yolo"] * use_yolo + ["contours"] + ["profondeur"] * (depth_fn is not None)
                                + ["mur"] * (depth_fn is not None and item_fn is not None)),
        "par_type": {lib: sum(1 for d in holes_details if d["etat"] == e) for e, (_, lib, _) in TYPES.items()},
        "nb_etiquettes_vues": n_labels,
        "rangees_ecartees": n_rows_dropped,
        "nb_trous_vus_grace_au_mur": n_wall_holes,
        "nb_produits_detectes": n_items,
        "preuves_detection": detection_proofs,
        "nb_anomalies": len(manifest),
        "total_boites_a_sortir": sum(m["quantite_a_reassortir"] for m in manifest),
        "duree_s": round(time.time() - t0, 2),
        "avertissements": warnings,
    }
    (out_dir / "audit_details.json").write_text(json.dumps({
        "resume": summary, "trous": holes_details, "images": frame_reports,
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    anomalies = [{**m, **d} for m, d in zip(manifest, holes_details)]
    _progress(1.0, "Terminé")
    return {"manifest": manifest, "manifest_path": str(manifest_path), "anomalies": anomalies,
            "annotated_path": str(annotated_path), "summary": summary}
