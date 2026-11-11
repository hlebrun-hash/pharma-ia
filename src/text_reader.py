"""Lecture du texte écrit sur les produits détectés et sur les étiquettes (OCR 100 % local).

Moteur : RapidOCR (modèles PP-OCR au format ONNX, livrés dans le paquet pip, aucun appel
réseau). Il lit les lettres latines majuscules / minuscules et les chiffres : suffisant pour
les noms de marque (DOLIPRANE, ADVIL…) et les dosages (1000 mg, 20 cp…).

Le texte lu sert à :
  * nommer chaque produit détecté (preuve + liste) ;
  * nommer l'emplacement à réassortir (produits restants de la même pile, ou étiquette) ;
  * retrouver le produit dans une base : ``data/base_produits.csv`` (colonne « nom », plus
    toute autre colonne utile : code CIP, référence…) ou, à défaut, les noms des dossiers de
    ``data/catalogue/``.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src import CATALOGUE_DIR, DATA_DIR

OCR_MIN_CONF = 0.6        # confiance minimale d'une ligne lue
OCR_MIN_CHARS = 3         # « MG », « 1 » seuls n'aident pas
OCR_TARGET_H = 320        # petite vignette agrandie pour que les lettres soient lisibles
MATCH_MIN_SCORE = 82      # similarité minimale (0-100) pour reconnaître un produit de la base
BASE_CSV = DATA_DIR / "base_produits.csv"


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def read_text(crop: np.ndarray) -> list[dict]:
    """Lignes de texte lues dans une vignette : [{"texte", "conf", "hauteur"}], de haut en bas."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
        return []
    h = crop.shape[0]
    if h < OCR_TARGET_H:
        f = min(4.0, OCR_TARGET_H / h)
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    result, _ = _engine()(crop)
    lines = []
    for box, text, conf in result or []:
        text = re.sub(r"\s+", " ", str(text)).strip()
        if float(conf) < OCR_MIN_CONF or sum(c.isalnum() for c in text) < OCR_MIN_CHARS:
            continue
        ys = [p[1] for p in box]
        lines.append({"texte": text, "conf": round(float(conf), 2),
                      "hauteur": round(max(ys) - min(ys), 1), "y": min(ys)})
    lines.sort(key=lambda l: l["y"])
    for l in lines:
        l.pop("y")
    return lines


def main_name(lines: list[dict]) -> str:
    """Nom le plus probable : la ligne écrite le plus gros et contenant des lettres
    (sur une boîte, la marque est presque toujours le plus grand texte)."""
    alpha = [l for l in lines if sum(c.isalpha() for c in l["texte"]) >= 3]
    if not alpha:
        return ""
    return max(alpha, key=lambda l: l["hauteur"] * l["conf"])["texte"]


def read_frame(img: np.ndarray) -> list[dict]:
    """Tout le texte de l'image en une seule passe : [{"texte", "conf", "hauteur", "box"}].
    Mesuré sur les vidéos de test : l'image entière se lit bien mieux (et ~7× plus vite)
    que boîte par boîte, car le détecteur de texte voit les lettres à leur vraie taille."""
    result, _ = _engine()(img)
    lines = []
    for box, text, conf in result or []:
        text = re.sub(r"\s+", " ", str(text)).strip()
        if float(conf) < OCR_MIN_CONF or sum(c.isalnum() for c in text) < OCR_MIN_CHARS:
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        lines.append({"texte": text, "conf": round(float(conf), 2),
                      "hauteur": round(max(ys) - min(ys), 1),
                      "box": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]})
    lines.sort(key=lambda l: (l["box"][1], l["box"][0]))
    return lines


def lines_in(lines: list[dict], box, pad: float = 0.04) -> list[dict]:
    """Lignes de texte dont le centre tombe dans le cadre (légèrement élargi)."""
    x1, y1, x2, y2 = box
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    out = []
    for l in lines:
        cx, cy = (l["box"][0] + l["box"][2]) / 2, (l["box"][1] + l["box"][3]) / 2
        if x1 - px <= cx <= x2 + px and y1 - py <= cy <= y2 + py:
            out.append(l)
    return out


def read_items(img: np.ndarray, items: list[dict], lines: list[dict] | None = None) -> list[dict]:
    """Ajoute à chaque produit détecté : item["texte"] (lignes lues) et item["nom_lu"].
    Renvoie toutes les lignes lues dans l'image."""
    if lines is None:
        lines = read_frame(img)
    for it in items:
        mine = lines_in(lines, it["bbox"])
        it["texte"] = [l["texte"] for l in mine]
        it["nom_lu"] = main_name(mine)
    return lines


def read_label(img: np.ndarray, bbox) -> list[str]:
    """Texte de l'étiquette de prix (nom du produit souvent écrit en petit au-dessus du prix)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    crop = img[int(max(0, y1 - 0.1 * h)):int(min(H, y2 + 0.1 * h)),
               int(max(0, x1 - 0.1 * w)):int(min(W, x2 + 0.1 * w))]
    return [l["texte"] for l in read_text(crop)]


LABEL_MIN_PX = 60   # en dessous de cette largeur (px), le texte d'une étiquette est illisible

_PRICE = re.compile(r"^\D{0,2}(\d{1,3})\s*[€E,.\s]?\s*(\d{2})\D{0,2}$")
_NOISE = re.compile(r"^(VOL|CONT|PRIX|EUR|LE\s?KG|LE\s?L|UNITE)\b", re.I)


def parse_label(lines: list[str]) -> dict:
    """Étiquette électronique (ESL) lue → {"nom", "prix"}.
    Format habituel : 1 à 2 lignes de nom en haut (« SEBIUM KERATO+ », « BODY SPRAY »),
    puis le prix en gros (« 18€90 », lu « 1890 »), puis des codes (EAN, contenance…)."""
    name_parts, price = [], None
    for t in lines:
        s = t.strip()
        compact = s.replace(" ", "")
        m = _PRICE.match(compact)
        digits = sum(c.isdigit() for c in compact)
        if price is None and m and len(compact) <= 7:
            price = f"{int(m.group(1))},{m.group(2)} €"
            continue
        letters = sum(c.isalpha() for c in s)
        if letters < 3 or digits > letters or _NOISE.match(s):
            continue
        if price is None and len(name_parts) < 2:   # le nom est écrit AU-DESSUS du prix
            name_parts.append(s)
    return {"nom": " ".join(name_parts).strip(), "prix": price}


def read_label_full(img_full: np.ndarray, bbox, scale: float = 1.0) -> dict:
    """Lit l'étiquette dans l'image en PLEINE résolution (bbox donnée dans l'image d'analyse,
    réduite d'un facteur `scale`). Renvoie {"nom", "prix", "lignes", "lisible"}."""
    x1, y1, x2, y2 = (v / scale for v in bbox)
    if x2 - x1 < LABEL_MIN_PX:      # vidéo trop petite / étiquette trop loin : inutile d'essayer
        return {"nom": "", "prix": None, "lignes": [], "lisible": False}
    lines = read_label(img_full, [x1, y1, x2, y2])
    return {**parse_label(lines), "lignes": lines, "lisible": True}


# ---------------------------------------------------------------------------- base produits

def _norm(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9 ]+", " ", s).strip()


@lru_cache(maxsize=1)
def load_base() -> tuple[dict, ...]:
    """Produits connus : data/base_produits.csv (séparateur ; ou ,) sinon data/catalogue/."""
    rows: list[dict] = []
    if BASE_CSV.exists():
        text = BASE_CSV.read_text(encoding="utf-8-sig")
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=";,\t")
        for r in csv.DictReader(text.splitlines(), dialect=dialect):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            if r.get("nom"):
                rows.append(r)
    elif Path(CATALOGUE_DIR).exists():
        rows = [{"nom": d.name} for d in sorted(Path(CATALOGUE_DIR).iterdir()) if d.is_dir()]
    return tuple(rows)


def match_base(texts: list[str]) -> dict | None:
    """Produit de la base le plus proche du texte lu, ou None si rien de convaincant.
    Renvoie la ligne de la base + "score" (0-100) et "texte_reconnu"."""
    from rapidfuzz import fuzz

    base = load_base()
    if not base or not texts:
        return None
    best, best_score, best_txt = None, 0.0, ""
    for t in texts:
        nt = _norm(t)
        if len(nt) < OCR_MIN_CHARS:
            continue
        for row in base:
            nb = _norm(row["nom"])
            brand = nb.split(" ")[0]
            # la marque (1er mot du nom en base) lue telle quelle sur la boîte compte beaucoup
            score = max(fuzz.token_set_ratio(nt, nb),
                        fuzz.ratio(brand, nt.split(" ")[0]) if len(brand) >= 4 else 0)
            if score > best_score:
                best, best_score, best_txt = row, score, t
    if best is None or best_score < MATCH_MIN_SCORE:
        return None
    return {**best, "score": round(best_score), "texte_reconnu": best_txt}


def identify(texts: list[str]) -> dict:
    """{"nom": nom à afficher, "source": "base" | "texte lu" | "", "base": ligne trouvée}."""
    m = match_base(texts)
    if m:
        return {"nom": m["nom"], "source": "base", "base": m}
    for t in texts:
        if sum(c.isalpha() for c in t) >= 3:
            return {"nom": t, "source": "texte lu", "base": None}
    return {"nom": "", "source": "", "base": None}
