"""Interface locale (Streamlit) — Réassort de rayon en pharmacie.

    streamlit run app.py

On envoie une photo ou une vidéo du rayon ; l'outil analyse UNIQUEMENT ces images (aucune
photo de référence du rayon plein) : il vérifie l'emplacement au-dessus de chaque étiquette
de prix et liste les produits manquants, avec une photo de preuve et une liste imprimable.
"""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import cv2
import streamlit as st

from src import CAPTURES_DIR, MODELS_DIR as ROOT_MODELS
from src.label_audit import run_label_audit
from src.shelf_detector import find_retail_weights

st.set_page_config(page_title="Réassort Pharmacie", page_icon="💊", layout="wide")

IMG_TYPES = ["jpg", "jpeg", "png", "webp"]
VID_TYPES = ["mp4", "mov", "avi", "mkv", "m4v", "3gp", "webm"]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return text or "rayon"


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def printable_checklist(manifest: list[dict], titre: str) -> str:
    rows = "".join(
        f"<tr><td class='box'>☐</td><td>{html.escape(m['produit'])}</td>"
        f"<td>{html.escape(m['emplacement'])}</td><td>{html.escape(m['type_anomalie'])}</td>"
        f"<td class='n'>{m['quantite_presente']}/{m['capacite_totale']}</td>"
        f"<td class='n big'>{m['quantite_a_reassortir']}</td></tr>"
        for m in manifest)
    total = sum(m["quantite_a_reassortir"] for m in manifest)
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>{html.escape(titre)}</title><style>
body{{font-family:Arial,sans-serif;margin:24px;color:#111}} h1{{font-size:20px;margin:0 0 4px}}
p{{margin:0 0 16px;color:#555}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #999;padding:8px;font-size:14px;text-align:left}} th{{background:#eee}}
.box{{font-size:20px;text-align:center;width:32px}} .n{{text-align:center}} .big{{font-weight:bold;font-size:16px}}
@media print{{button{{display:none}}}}</style></head><body>
<h1>{html.escape(titre)}</h1><p>{time.strftime('%d/%m/%Y %H:%M')} — {len(manifest)} ligne(s), {total} boîte(s) à sortir de la réserve</p>
<button onclick="window.print()">Imprimer</button>
<table><tr><th>✓</th><th>Produit</th><th>Emplacement</th><th>Anomalie</th><th>En rayon</th><th>À sortir</th></tr>{rows}</table>
</body></html>"""


# --------------------------------------------------------------------------------------
# Barre latérale : réglages
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Réglages")
    weights_found = find_retail_weights()
    generic_yolo = (ROOT_MODELS / "yolov8m.pt").exists()
    detector = st.selectbox(
        "Détecteur de boîtes", ["auto", "classical", "yolo"],
        help="auto = YOLO « rayon » (ex. models/sku110k.pt) s'il existe, sinon YOLO générique "
             "(models/yolov8m.pt, détecte les flacons), sinon détection OpenCV classique.")
    if weights_found:
        st.caption(f"Poids YOLO rayon : `{Path(weights_found).name}`")
    elif generic_yolo:
        st.caption("YOLO générique `yolov8m.pt` : bon sur les flacons, rate souvent "
                   "tubes, boîtes et sachets.")
    else:
        st.caption("Aucun poids YOLO dans `models/` → détection classique (rayons simples).")
    conf = st.slider("Confiance YOLO", 0.05, 0.9, 0.25, 0.05)
    interval = st.slider("Vidéo : 1 image nette toutes les (s)", 0.5, 4.0, 1.5, 0.5)

st.title("💊 Réassort de rayon — Pharmacie")
st.caption("Analyse 100 % locale : aucune donnée ne quitte cet ordinateur.")

# --------------------------------------------------------------------------------------
# Trous au-dessus des étiquettes de prix (aucune référence : seulement les images envoyées)
# --------------------------------------------------------------------------------------
with st.container():
    st.subheader("Filmer le rayon : on vérifie l'emplacement au-dessus de chaque étiquette de prix")
    st.caption("🟥 vide : aucun produit au-dessus de l'étiquette · 🟧 espace : on voit le mur "
               "entre les produits → un produit manque · 🟦 pile incomplète : des produits "
               "à l'avant mais le mur visible au-dessus.  \nAvec « chaque produit » coché, les "
               "trous sans étiquette électronique sont aussi cherchés grâce au mur (profondeur + "
               "surface unie).  \nFilmez en 4K ou 1080p, de face, à environ 1 m, lentement "
               "(chaque étiquette ≈ 3 s à l'écran). Aucune photo de référence n'est nécessaire.")
    use_depth = st.checkbox("Regarder aussi entre l'avant du rayon et le mur (profondeur)", value=True,
                            help="Modèle local Depth Anything V2 (~1,5 s par image). Décochez pour "
                                 "ne chercher que les emplacements vides.")
    read_txt = st.checkbox("Lire le nom du produit sur l'étiquette de prix (OCR) et le chercher "
                           "dans la base", value=True,
                           help="RapidOCR, 100 % local, sur l'image en pleine résolution : filmez en "
                                "4K ou en 1080p, à environ 1 m. Base : data/base_produits.csv "
                                "(colonne « nom ») ou, à défaut, les dossiers de data/catalogue/.")
    all_items = st.checkbox("Détecter et numéroter chaque produit (modèle SKU-110K)", value=False,
                            help="YOLO11s spécialisé rayon (SKU-110K), 100 % local et rapide (~0,2 s par image). "
                                 "Numérote chaque produit visible, ajoute les espaces entre produits et les trous sans étiquette.")
    cap_t = st.file_uploader("Photo ou vidéo du rayon", type=IMG_TYPES + VID_TYPES, key="trous_upload")
    if cap_t is not None and st.button("🔍 Chercher les produits manquants", type="primary"):
        dest = CAPTURES_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{slugify(Path(cap_t.name).stem)}{Path(cap_t.name).suffix.lower()}"
        dest.write_bytes(cap_t.getbuffer())
        bar = st.progress(0.0, text="Préparation…")
        try:
            st.session_state["trous"] = run_label_audit(
                str(dest), interval=interval, detector=detector, conf=conf, use_depth=use_depth,
                detect_all_items=all_items, read_text=read_txt, progress=lambda f, m: bar.progress(f, text=m))
            bar.empty()
        except Exception as e:
            st.session_state.pop("trous", None)
            st.error(f"L'analyse a échoué : {e}")

    rt = st.session_state.get("trous")
    if rt:
        s = rt["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Emplacements à réassortir", s["nb_anomalies"])
        m2.metric("Produits détectés", s.get("nb_produits_detectes", 0),
                  help="Total sur toutes les images : un même produit vu sur 2 images compte 2 fois.")
        m3.metric("Étiquettes vérifiées", s["nb_etiquettes_vues"])
        m4.metric("Images analysées", len(s["frames_analysees"]))
        if s.get("par_type"):
            st.caption(" · ".join(f"{lib} : **{n}**" for lib, n in s["par_type"].items()))
        for w in s["avertissements"]:
            st.warning(w)
        if not rt["manifest"]:
            st.success("✅ Aucun trou ni produit manquant détecté.")
        else:
            st.subheader("📋 Emplacements à réassortir")
            icons = {"vide": "🟥", "espace": "🟧", "incomplet": "🟦", "retrait": "🟨"}
            for a in rt["anomalies"]:
                c1, c2 = st.columns([2, 3])
                if a.get("image_preuve") and Path(a["image_preuve"]).exists():
                    c1.image(a["image_preuve"], width="stretch",
                             caption="cadre coloré = zone concernée · 🟪 étiquette de prix")
                c2.markdown(f"### {icons.get(a.get('etat'), '🟥')} n°{a['id']} — "
                            f"{a.get('libelle', 'Emplacement vide')}\n{a['emplacement']}  \n"
                            f"Vu sur **{a['nb_vues']} image(s)** — certitude **{a['certitude']}**"
                            + (f"  \n📦 **{a['quantite_a_reassortir']} boîte(s)** à remettre"
                               if a.get("quantite_a_reassortir", 1) > 1 else "")
                            + ("  \n🧱 trouvé grâce au mur visible (sans étiquette)"
                               if a.get("source") == "mur" else ""))
                idt = a.get("identification") or {}
                if idt.get("source") == "base":
                    extra = " · ".join(f"{k} : {v}" for k, v in (idt.get("base") or {}).items()
                                       if k not in ("nom", "score", "texte_reconnu") and v)
                    c2.markdown(f"💊 **{idt['nom']}** — reconnu dans la base "
                                f"(lu : « {idt['base'].get('texte_reconnu', '')} », "
                                f"via {idt.get('via') or 'texte'})" + (f"  \n{extra}" if extra else ""))
                elif idt.get("nom"):
                    src_txt = "sur l'étiquette" if idt.get("via") == "etiquette" else "sur les produits"
                    c2.markdown(f"💊 Lu {src_txt} : **{idt['nom']}**"
                                + (f" — {idt['prix']}" if idt.get("prix") else "")
                                + " (pas trouvé dans la base, à vérifier)")
                elif idt.get("voisins"):
                    c2.markdown(f"💊 Nom non lu — à côté de : {', '.join(idt['voisins'])}")
                if a.get("preuve_detection") and Path(a["preuve_detection"]).exists():
                    with c2.popover("🔎 Voir tous les produits détectés sur cette image"):
                        st.image(a["preuve_detection"], width="stretch")
                c2.checkbox("✓ Rempli", key=f"trou_fait_{a['id']}")
            d1, d2 = st.columns(2)
            d1.download_button("🖨️ Liste imprimable (HTML)",
                               printable_checklist(rt["manifest"], "Emplacements à réassortir").encode("utf-8"),
                               "trous_a_remplir.html", "text/html")
            d2.download_button("⬇️ Manifeste JSON",
                               json.dumps(rt["manifest"], indent=2, ensure_ascii=False).encode("utf-8"),
                               "reassort_manifest.json", "application/json")
        proofs = [p for p in s.get("preuves_detection", []) if Path(p).exists()]
        if proofs:
            st.subheader("🔎 Preuve : chaque produit détecté, image par image")
            st.caption("Bleu clair = un produit détecté (numéroté) · violet = étiquette de prix, avec "
                       "le nombre de produits vus au-dessus · vert translucide = mur visible · "
                       "couleurs pleines = anomalies.")
            k = st.slider("Image", 1, len(proofs), 1, key="proof_idx") if len(proofs) > 1 else 1
            st.image(proofs[k - 1], width="stretch", caption=f"Image {k}/{len(proofs)}")
        with st.expander("🖼️ Vue d'ensemble des anomalies (🟩 occupé, 🟥 vide, 🟧 espace, 🟦 pile incomplète)"):
            st.image(bgr_to_rgb(cv2.imread(rt["annotated_path"])), width="stretch")
