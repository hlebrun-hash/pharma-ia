"""Orchestrateur — audit de réassort d'un rayon de pharmacie.

Usage :
    python src/pipeline.py --ref data/references/rayon_01_plein.jpg \
                           --input data/captures/visite_rayon_01.mp4

Produit :
    data/outputs/reassort_manifest.json   (liste des produits à réassortir)
    data/outputs/audit_details.json       (diagnostic complet)
    data/outputs/audit_annote.jpg         (image annotée : rouge = rupture, orange = partiel)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pydantic import BaseModel, Field, TypeAdapter  # noqa: E402

from src import OUTPUTS_DIR  # noqa: E402
from src.local_vlm import identify_product_from_crop  # noqa: E402
from src.planogram_matcher import (  # noqa: E402
    annotate,
    compare_with_reference,
    load_or_build_planogram,
    save_planogram,
)
from src.video_engine import extract_sharp_frames, is_video  # noqa: E402


class ManifestItem(BaseModel):
    """Schéma obligatoire d'une ligne de reassort_manifest.json."""

    id: int = Field(ge=1)
    produit: str = Field(min_length=1)
    emplacement: str
    type_anomalie: str
    quantite_presente: int = Field(ge=0)
    capacite_totale: int = Field(ge=1)
    quantite_a_reassortir: int = Field(ge=1)
    image_preuve: str | None


MANIFEST_SCHEMA = TypeAdapter(list[ManifestItem])


def _rel(p: str | None) -> str | None:
    if not p:
        return p
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def name_products(planogram: dict, block_ids: set[int], use_vlm: bool = True) -> bool:
    """Identifie (une seule fois, résultat mis en cache dans le planogramme) les produits."""
    changed = False
    for b in planogram["blocks"]:
        if b["block_id"] not in block_ids or b.get("produit"):
            continue
        name, src = "", "inconnu"
        if b.get("crop_ref") and Path(b["crop_ref"]).exists():
            name, src = identify_product_from_crop(b["crop_ref"], use_vlm=use_vlm, return_source=True)
        if name:
            b["produit"], b["produit_source"] = name, src
            changed = True
    return changed


def run_audit(ref: str, input_path: str, detector: str = "auto", weights: str | None = None,
              conf: float = 0.25, interval: float = 1.5, use_vlm: bool = True,
              rebuild_planogram: bool = False, out_dir: str | Path = OUTPUTS_DIR,
              manifest_name: str = "reassort_manifest.json") -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # 1. Captures
    if is_video(input_path):
        frames = extract_sharp_frames(input_path, interval_sec=interval)
        if not frames:
            raise ValueError("Aucune frame exploitable extraite de la vidéo.")
    else:
        frames = [str(input_path)]

    # 2. Planogramme de référence
    planogram = load_or_build_planogram(ref, rebuild=rebuild_planogram, method=detector,
                                        weights=weights, conf=conf)

    # 3. Comparaison
    anomalies, details = compare_with_reference(ref, frames, planogram=planogram,
                                                method=detector, weights=weights, conf=conf,
                                                return_details=True)
    if not any(fr["alignment"]["method"] != "failed" for fr in details["frame_results"]):
        warnings.append("Impossible d'aligner la capture sur la photo de référence : "
                        "vérifiez qu'il s'agit bien du même rayon.")
    if any(fr["alignment"]["method"] == "identity-fallback" for fr in details["frame_results"]):
        warnings.append("Alignement approximatif (cadrage supposé identique à la référence).")
    unseen = sum(1 for s in details["slot_state"].values() if s == "non_vu")
    if unseen:
        warnings.append(f"{unseen} emplacement(s) hors champ de la capture : non audités.")

    # 4. Identification des produits concernés (VLM local / catalogue), mise en cache
    if name_products(planogram, {a["block_id"] for a in anomalies}, use_vlm=use_vlm):
        save_planogram(planogram)
    names = {b["block_id"]: b.get("produit") for b in planogram["blocks"]}

    # 5. Manifeste
    anomalies.sort(key=lambda a: (a["etagere"], a["colonne"]))
    manifest = []
    for k, a in enumerate(anomalies, start=1):
        a["id"] = k
        a["produit"] = names.get(a["block_id"]) or f"Produit non identifié (Ét.{a['etagere']} Col.{a['colonne']})"
        manifest.append({
            "id": k,
            "produit": a["produit"],
            "emplacement": f"Étagère {a['etagere']}, Colonne {a['colonne']}",
            "type_anomalie": a["type_anomalie"],
            "quantite_presente": a["quantite_presente"],
            "capacite_totale": a["capacite_totale"],
            "quantite_a_reassortir": a["quantite_a_reassortir"],
            "image_preuve": _rel(a["image_preuve"]),
        })
    MANIFEST_SCHEMA.validate_python(manifest)
    manifest_path = out_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. Image annotée
    frs = [fr for fr in details["frame_results"] if fr["alignment"]["H"] is not None]
    if len(frames) == 1 and frs:
        img = cv2.imread(frames[0])
        H_inv = np.linalg.inv(frs[0]["alignment"]["H"])
        annotated = annotate(img, planogram, anomalies, H_ref_to_img=H_inv, gaps=frs[0]["gaps"])
    else:  # vidéo : on reporte les zones sur la photo du rayon plein
        annotated = annotate(cv2.imread(ref), planogram, anomalies)
    annotated_path = out_dir / "audit_annote.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    summary = {
        "reference": _rel(ref),
        "capture": _rel(input_path),
        "frames_analysees": [_rel(f) for f in frames],
        "detecteur": planogram["detector"],
        "nb_emplacements_reference": len(planogram["slots"]),
        "nb_blocs_produit": len(planogram["blocks"]),
        "nb_anomalies": len(manifest),
        "total_boites_a_sortir": sum(m["quantite_a_reassortir"] for m in manifest),
        "duree_s": round(time.time() - t0, 2),
        "avertissements": warnings,
    }
    (out_dir / "audit_details.json").write_text(json.dumps({
        "resume": summary,
        "frames": details["frames"],
        "anomalies": [{k: v for k, v in a.items()} for a in anomalies],
        "trous_detectes": [g for fr in frs for g in fr["gaps"]],
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    return {"manifest": manifest, "manifest_path": str(manifest_path), "anomalies": anomalies,
            "annotated_path": str(annotated_path), "summary": summary, "planogram": planogram}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit de réassort d'un rayon de pharmacie (100 % local).")
    p.add_argument("--ref", default=None,
                   help="Photo du rayon plein (référence). Sans --ref : mode « étiquettes ».")
    p.add_argument("--input", required=True, help="Photo ou vidéo du rayon à auditer")
    p.add_argument("--detector", default="auto", choices=["auto", "yolo", "classical"])
    p.add_argument("--weights", default=None, help="Poids YOLO (.pt/.onnx), ex. models/sku110k.pt")
    p.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance YOLO")
    p.add_argument("--interval", type=float, default=1.5, help="Fenêtre (s) de sélection des frames")
    p.add_argument("--no-vlm", action="store_true", help="Ne pas interroger Ollama")
    p.add_argument("--no-depth", action="store_true",
                   help="Mode étiquettes : ne pas utiliser la profondeur (vides seulement)")
    p.add_argument("--items", action="store_true",
                   help="Mode étiquettes : détecter chaque produit (Grounding DINO, TRÈS lent sans "
                        "carte graphique ; ajoute les trous sans étiquette)")
    p.add_argument("--no-items", action="store_true", help=argparse.SUPPRESS)  # ancien défaut
    p.add_argument("--no-ocr", action="store_true",
                   help="Mode étiquettes : ne pas lire le nom du produit (étiquette, boîtes)")
    p.add_argument("--rebuild-planogram", action="store_true", help="Recalculer le planogramme")
    p.add_argument("--out-dir", default=str(OUTPUTS_DIR))
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not a.ref:
        from src.label_audit import run_label_audit
        r = run_label_audit(a.input, interval=a.interval, detector=a.detector, weights=a.weights,
                            conf=a.conf, use_depth=not a.no_depth,
                            detect_all_items=a.items and not a.no_items, read_text=not a.no_ocr,
                            out_dir=a.out_dir,
                            progress=lambda f, m: print(f"  [{f * 100:3.0f} %] {m}", flush=True))
        s = r["summary"]
        print(f"Mode sans référence | {s['nb_produits_detectes']} produit(s) détectés | "
              f"{s['nb_etiquettes_vues']} étiquette(s) vues sur "
              f"{len(s['frames_analysees'])} image(s) | détecteur : {s['detecteur']}")
        for w in s["avertissements"]:
            print(f"⚠ {w}")
        if not r["manifest"]:
            print("✔ Aucun trou ni produit manquant détecté.")
        for lib, n in s["par_type"].items():
            print(f"  {lib} : {n}")
        if s.get("nb_trous_vus_grace_au_mur"):
            print(f"  dont trouvés grâce au mur (sans étiquette) : {s['nb_trous_vus_grace_au_mur']}")
        for m, an in zip(r["manifest"], r["anomalies"]):
            qty = f", {m['quantite_a_reassortir']} boîte(s)" if m["quantite_a_reassortir"] > 1 else ""
            via = ", vu grâce au mur" if an.get("source") == "mur" else ""
            print(f"  [{m['id']:>2}] {an['libelle'].upper()} — {m['emplacement']} (certitude {an['certitude']}, "
                  f"vu sur {an['nb_vues']} image(s){qty}{via}) → {m['image_preuve']}")
            print(f"       Produit : {m['produit']}")
        if s["preuves_detection"]:
            print(f"Preuves (chaque produit numéroté) : {Path(s['preuves_detection'][0]).parent}")
        print(f"Manifeste : {r['manifest_path']}")
        print(f"Image annotée : {r['annotated_path']}")
        return 0

    r = run_audit(a.ref, a.input, detector=a.detector, weights=a.weights, conf=a.conf,
                  interval=a.interval, use_vlm=not a.no_vlm,
                  rebuild_planogram=a.rebuild_planogram, out_dir=a.out_dir)
    s = r["summary"]
    print(f"Détecteur : {s['detecteur']} | {s['nb_emplacements_reference']} emplacements, "
          f"{s['nb_blocs_produit']} blocs produit | {len(s['frames_analysees'])} capture(s)")
    for w in s["avertissements"]:
        print(f"⚠ {w}")
    if not r["manifest"]:
        print("✔ Rayon complet : rien à réassortir.")
    for m in r["manifest"]:
        print(f"  [{m['id']:>2}] {m['emplacement']:<24} {m['type_anomalie']:<19} "
              f"{m['quantite_presente']}/{m['capacite_totale']} → sortir {m['quantite_a_reassortir']} × {m['produit']}")
    print(f"Total à sortir de la réserve : {s['total_boites_a_sortir']} boîte(s)")
    print(f"Manifeste : {r['manifest_path']}\nImage annotée : {r['annotated_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
