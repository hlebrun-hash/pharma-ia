"""Mesure du taux de bons résultats sur des vidéos dont on connaît les vrais trous.

Vérité terrain : data/verite_terrain/<nom>.json
{
  "images": "data/captures/frames_<video>",          # images déjà extraites de la vidéo
  "taille": [largeur, hauteur],                       # taille des images annotées (optionnel)
  "trous": [{"id": "...", "nom": "nom attendu (optionnel)",
             "vues": {"<image>.jpg": [x1, y1, x2, y2], ...}}],   # zone où le produit manque
  "ignorer": {"<image>.jpg": [[x1, y1, x2, y2], ...]}           # cas ambigus (non comptés)
}

Pour chaque alerte de l'outil (un trou dédoublonné sur la vidéo) :
  JUSTE    si une de ses vues recouvre un vrai trou,
  DOUBLON  si ce vrai trou a déjà été signalé par une autre alerte,
  IGNORÉE  si elle ne tombe que dans une zone ambiguë,
  FAUSSE   sinon.
Précision = justes / (justes + fausses + doublons) ; rappel = vrais trous trouvés / vrais trous.

    python scripts/evaluer.py                      # toutes les vérités terrain
    python scripts/evaluer.py video_teste --no-depth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GT_DIR = ROOT / "data" / "verite_terrain"
MATCH = 0.3   # recouvrement minimal (intersection / plus petite surface)


def overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / m if m > 0 else 0.0


def _scale(box, sx, sy):
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def _norm(s: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9]+", " ", s).strip()


def score(gt: dict, anomalies: list[dict], sizes: dict[str, tuple[int, int]]) -> dict:
    """Compare les alertes (avec leurs vues) à la vérité terrain."""
    tw, th = gt.get("taille") or (None, None)

    def gt_box(img, box):
        if not tw or img not in sizes:
            return box
        w, h = sizes[img]
        return _scale(box, w / tw, h / th)

    holes = {h["id"]: h for h in gt.get("trous", [])}
    found: dict[str, int] = {}
    rows = []
    for a in anomalies:
        status, hole_id = "FAUSSE", None
        for v in a.get("vu_sur_images", []):
            img = Path(v["image"]).name
            z = v.get("zone") or a["zone"]
            for hid, h in holes.items():
                if img in h["vues"] and overlap(z, gt_box(img, h["vues"][img])) >= MATCH:
                    status, hole_id = "JUSTE", hid
                    break
            if status == "JUSTE":
                break
            if any(overlap(z, gt_box(img, b)) >= MATCH for b in gt.get("ignorer", {}).get(img, [])):
                status = "IGNORÉE"
        if status == "JUSTE" and hole_id in found:
            status = "DOUBLON"
        if status == "JUSTE":
            found[hole_id] = a.get("id", 0)
        name_ok = None
        if status == "JUSTE" and holes[hole_id].get("nom"):
            want = _norm(holes[hole_id]["nom"])
            got = _norm(a.get("produit", ""))
            name_ok = bool(want) and want in got
        rows.append({"id": a.get("id"), "statut": status, "trou": hole_id, "etat": a.get("etat"),
                     "produit": a.get("produit", ""), "nom_ok": name_ok,
                     "vues": [Path(v["image"]).name for v in a.get("vu_sur_images", [])]})
    n_tp = sum(r["statut"] == "JUSTE" for r in rows)
    n_fp = sum(r["statut"] == "FAUSSE" for r in rows)
    n_dup = sum(r["statut"] == "DOUBLON" for r in rows)
    names = [r["nom_ok"] for r in rows if r["nom_ok"] is not None]
    return {"alertes": rows, "justes": n_tp, "fausses": n_fp, "doublons": n_dup,
            "ignorees": sum(r["statut"] == "IGNORÉE" for r in rows),
            "vrais_trous": len(holes), "trouves": len(found),
            "manques": sorted(set(holes) - set(found)),
            "noms_justes": sum(names), "noms_evalues": len(names)}


def run_one(gt_path: Path, args) -> dict | None:
    from src.label_audit import run_label_audit
    import cv2

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    img_dir = ROOT / gt["images"]
    if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
        print(f"[{gt_path.stem}] images absentes ({gt['images']}) : ignoré")
        return None
    out = ROOT / "data" / "outputs" / "evaluation" / gt_path.stem
    t0 = time.time()
    r = run_label_audit(str(img_dir), use_depth=not args.no_depth,
                        detect_all_items=args.items, read_text=not args.no_ocr, out_dir=out)
    dt = time.time() - t0
    sizes = {}
    for f in img_dir.glob("*.jpg"):
        im = cv2.imread(str(f))
        if im is not None:
            sizes[f.name] = (im.shape[1], im.shape[0])
    res = score(gt, r["anomalies"], sizes)
    res.update({"video": gt_path.stem, "duree_s": round(dt, 1), "nb_images": len(r["summary"]["frames_analysees"])})
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("noms", nargs="*", help="vérités terrain à évaluer (défaut : toutes)")
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--items", action="store_true", help="détecter chaque produit (lent)")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--json", default=None, help="écrire le détail dans ce fichier")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    paths = sorted(GT_DIR.glob("*.json"))
    if args.noms:
        paths = [p_ for p_ in paths if p_.stem in args.noms]
    results = [r for r in (run_one(p_, args) for p_ in paths) if r]
    tot = {k: sum(r[k] for r in results) for k in ("justes", "fausses", "doublons", "ignorees",
                                                   "vrais_trous", "trouves", "noms_justes", "noms_evalues")}
    for r in results:
        print(f"\n=== {r['video']} : {r['nb_images']} image(s) en {r['duree_s']} s")
        for a in r["alertes"]:
            print(f"  [{a['statut']:<8}] #{a['id']} {a['etat']:<10} {a['produit'][:60]!r} vues={a['vues']}")
        print(f"  vrais trous trouvés : {r['trouves']}/{r['vrais_trous']}  manqués : {r['manques']}")
    alerts = tot["justes"] + tot["fausses"] + tot["doublons"]
    prec = tot["justes"] / alerts if alerts else 1.0
    rec = tot["trouves"] / tot["vrais_trous"] if tot["vrais_trous"] else 1.0
    print("\n=== TOTAL")
    print(f"  alertes comptées : {alerts} (justes {tot['justes']}, fausses {tot['fausses']}, "
          f"doublons {tot['doublons']}) + {tot['ignorees']} ambiguë(s) non comptée(s)")
    print(f"  PRÉCISION (alertes justes) : {prec:.0%}")
    print(f"  RAPPEL (vrais trous trouvés) : {rec:.0%}  ({tot['trouves']}/{tot['vrais_trous']})")
    if tot["noms_evalues"]:
        print(f"  NOMS JUSTES : {tot['noms_justes']}/{tot['noms_evalues']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"total": tot, "videos": results}, indent=2,
                                              ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
