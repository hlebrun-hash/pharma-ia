"""Interface locale (Streamlit) — Réassort de rayon en pharmacie.

    streamlit run app.py

Onglet 1 : trous dans le rayon (étiquettes de prix), sans aucune photo de référence.
Onglet 2 : photo du rayon plein (une seule fois) + correction des noms de produits.
Onglet 3 : audit du jour comparé à la référence → zones à remplir + checklist imprimable.
"""

from __future__ import annotations

import base64
import html
import json
import re
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src import CAPTURES_DIR, MODELS_DIR as ROOT_MODELS, REFERENCES_DIR
from src.local_vlm import DEFAULT_VLM_MODEL, get_catalog, identify_product_from_crop, ollama_available
from src.planogram_matcher import annotate_planogram, build_planogram, planogram_path, save_planogram
from src.label_audit import run_label_audit
from src.pipeline import run_audit
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


def list_references() -> list[Path]:
    return sorted(p for p in REFERENCES_DIR.iterdir()
                  if p.suffix.lower().lstrip(".") in IMG_TYPES)


def load_planogram(ref: Path) -> dict | None:
    p = planogram_path(ref)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thumb_data_uri(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


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
    use_vlm = st.checkbox("Identifier les produits avec le VLM local (Ollama)", value=True)
    if use_vlm:
        if ollama_available(DEFAULT_VLM_MODEL):
            st.success(f"Ollama prêt ({DEFAULT_VLM_MODEL})")
        else:
            st.info(f"Ollama / `{DEFAULT_VLM_MODEL}` non détecté → catalogue visuel "
                    "(`data/catalogue/`) puis saisie manuelle.")
    interval = st.slider("Vidéo : 1 image nette toutes les (s)", 0.5, 4.0, 1.5, 0.5)

st.title("💊 Réassort de rayon — Pharmacie")
st.caption("Analyse 100 % locale : aucune donnée ne quitte cet ordinateur.")

tab_trous, tab_ref, tab_audit = st.tabs(["① Trous dans le rayon (étiquettes)",
                                         "② Rayon plein (référence)",
                                         "③ Audit comparé à la référence"])

# --------------------------------------------------------------------------------------
# Onglet 1 : trous au-dessus des étiquettes de prix (sans référence)
# --------------------------------------------------------------------------------------
with tab_trous:
    st.subheader("Filmer le rayon : on vérifie l'emplacement au-dessus de chaque étiquette de prix")
    st.caption("🟥 vide : aucun produit au-dessus de l'étiquette · 🟧 espace : on voit le mur "
               "entre les produits → un produit manque · 🟨 retrait : la façade est loin derrière "
               "l'étiquette → les boîtes de devant manquent · 🟦 pile incomplète : des produits "
               "à l'avant mais le mur visible au-dessus.  \nFilmez de face, à environ 1 m, "
               "lentement, étiquettes bien visibles. Aucune photo de référence n'est nécessaire.")
    use_depth = st.checkbox("Regarder aussi entre l'avant du rayon et le mur (profondeur)", value=True,
                            help="Modèle local Depth Anything V2 (~1,5 s par image). Décochez pour "
                                 "ne chercher que les emplacements vides.")
    all_items = st.checkbox("Détecter et numéroter CHAQUE produit (preuve image par image)", value=True,
                            help="Grounding DINO + YOLO, 100 % local. Lent : environ 1,5 min par "
                                 "image la première fois (résultat gardé en mémoire ensuite). "
                                 "Permet aussi de trouver les espaces entre deux produits.")
    read_txt = st.checkbox("Lire le nom écrit sur les produits (OCR) et le chercher dans la base",
                           value=True, disabled=not all_items,
                           help="RapidOCR, 100 % local. Base : data/base_produits.csv (colonne « nom ») "
                                "ou, à défaut, les dossiers de data/catalogue/.")
    cap_t = st.file_uploader("Photo ou vidéo du rayon", type=IMG_TYPES + VID_TYPES, key="trous_upload")
    if cap_t is not None and st.button("🔍 Chercher les produits manquants", type="primary"):
        dest = CAPTURES_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{slugify(Path(cap_t.name).stem)}{Path(cap_t.name).suffix.lower()}"
        dest.write_bytes(cap_t.getbuffer())
        bar = st.progress(0.0, text="Préparation…")
        try:
            st.session_state["trous"] = run_label_audit(
                str(dest), interval=interval, detector=detector, conf=conf, use_depth=use_depth,
                detect_all_items=all_items, read_text=all_items and read_txt, progress=lambda f, m: bar.progress(f, text=m))
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
                            f"Vu sur **{a['nb_vues']} image(s)** — certitude **{a['certitude']}**")
                idt = a.get("identification") or {}
                if idt.get("source") == "base":
                    extra = " · ".join(f"{k} : {v}" for k, v in (idt.get("base") or {}).items()
                                       if k not in ("nom", "score", "texte_reconnu") and v)
                    c2.markdown(f"💊 **{idt['nom']}** — reconnu dans la base "
                                f"(lu : « {idt['base'].get('texte_reconnu', '')} », "
                                f"via {idt.get('via') or 'texte'})" + (f"  \n{extra}" if extra else ""))
                elif idt.get("nom"):
                    c2.markdown(f"💊 Texte lu : **{idt['nom']}** (pas trouvé dans la base, à vérifier)")
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
                       "le nombre de produits vus au-dessus · couleurs pleines = anomalies.")
            k = st.slider("Image", 1, len(proofs), 1, key="proof_idx") if len(proofs) > 1 else 1
            st.image(proofs[k - 1], width="stretch", caption=f"Image {k}/{len(proofs)}")
        with st.expander("🖼️ Vue d'ensemble des anomalies (🟩 occupé, 🟥 vide, 🟧 espace, 🟦 pile incomplète, 🟨 retrait)"):
            st.image(bgr_to_rgb(cv2.imread(rt["annotated_path"])), width="stretch")

# --------------------------------------------------------------------------------------
# Onglet 1 : référence
# --------------------------------------------------------------------------------------
with tab_ref:
    st.subheader("Photographier le rayon bien rangé et complet (une seule fois)")
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Photo du rayon plein", type=IMG_TYPES, key="ref_upload")
    with c2:
        rayon_name = st.text_input("Nom du rayon", value="rayon_01_plein")
        sim_thr = st.slider("Regroupement des boîtes identiques", 0.3, 0.9, 0.6, 0.05,
                            help="Plus haut = blocs produit plus petits / plus stricts.")
    if up is not None and st.button("💾 Enregistrer et analyser la référence", type="primary"):
        ext = Path(up.name).suffix.lower() or ".jpg"
        dest = REFERENCES_DIR / f"{slugify(rayon_name)}{ext}"
        dest.write_bytes(up.getbuffer())
        with st.spinner("Détection des emplacements…"):
            try:
                pl = build_planogram(str(dest), method=detector, conf=conf, similarity_threshold=sim_thr)
                save_planogram(pl)
                st.session_state["selected_ref"] = dest.name
                st.success(f"{len(pl['slots'])} boîtes détectées, regroupées en "
                           f"{len(pl['blocks'])} blocs produit.")
            except Exception as e:
                st.error(f"Analyse impossible : {e}")

    refs = list_references()
    if refs:
        st.divider()
        names = [p.name for p in refs]
        default = names.index(st.session_state["selected_ref"]) if st.session_state.get("selected_ref") in names else 0
        sel = st.selectbox("Référence à consulter / corriger", names, index=default, key="ref_view")
        ref_path = REFERENCES_DIR / sel
        pl = load_planogram(ref_path)
        if pl is None:
            if st.button("Analyser cette référence"):
                pl = build_planogram(str(ref_path), method=detector, conf=conf, similarity_threshold=sim_thr)
                save_planogram(pl)
                st.rerun()
        else:
            col_img, col_tab = st.columns([3, 2])
            with col_img:
                st.image(bgr_to_rgb(annotate_planogram(cv2.imread(str(ref_path)), pl)),
                         caption=f"Blocs produit détectés (détecteur : {pl['detector']}) — "
                                 "E = étagère, C = colonne, x = capacité", width="stretch")
                if st.button("🔁 Recalculer les blocs avec les réglages actuels"):
                    new = build_planogram(str(ref_path), method=detector, conf=conf, similarity_threshold=sim_thr)
                    save_planogram(new)
                    st.rerun()
            with col_tab:
                df = pd.DataFrame([{
                    "Bloc": b["block_id"], "Vignette": thumb_data_uri(b.get("crop_ref")),
                    "Étagère": b["etagere"], "Colonne": b["colonne"], "Capacité": b["capacite"],
                    "Produit": b.get("produit") or "",
                } for b in pl["blocks"]])
                edited = st.data_editor(
                    df, hide_index=True, width="stretch", key=f"edit_{sel}",
                    disabled=["Bloc", "Vignette", "Étagère", "Colonne", "Capacité"],
                    column_config={"Vignette": st.column_config.ImageColumn("Vignette", width="small"),
                                   "Produit": st.column_config.TextColumn("Produit (modifiable)")})
                b1, b2 = st.columns(2)
                if b1.button("💾 Enregistrer les noms"):
                    by_id = {int(r["Bloc"]): str(r["Produit"]).strip() for _, r in edited.iterrows()}
                    for b in pl["blocks"]:
                        name = by_id.get(b["block_id"], "")
                        if name != (b.get("produit") or ""):
                            b["produit"] = name or None
                            b["produit_source"] = "manuel" if name else None
                    save_planogram(pl, planogram_path(ref_path))
                    st.success("Noms enregistrés.")
                if b2.button("🤖 Identifier automatiquement"):
                    get_catalog(rebuild=True)
                    with st.spinner("Identification locale des produits…"):
                        n = 0
                        for b in pl["blocks"]:
                            if b.get("produit") or not b.get("crop_ref"):
                                continue
                            name, src = identify_product_from_crop(b["crop_ref"], use_vlm=use_vlm,
                                                                   return_source=True)
                            if name:
                                b["produit"], b["produit_source"] = name, src
                                n += 1
                        save_planogram(pl, planogram_path(ref_path))
                    st.success(f"{n} produit(s) identifié(s).")
                    st.rerun()
    else:
        st.info("Aucune référence enregistrée pour l'instant.")

# --------------------------------------------------------------------------------------
# Onglet 2 : audit
# --------------------------------------------------------------------------------------
with tab_audit:
    refs = list_references()
    if not refs:
        st.warning("Commencez par enregistrer la photo du rayon plein (onglet ②).")
    else:
        names = [p.name for p in refs]
        ref_sel = st.selectbox("Rayon audité (référence)", names, key="ref_audit")
        cap = st.file_uploader("Photo ou vidéo du rayon aujourd'hui", type=IMG_TYPES + VID_TYPES,
                               key="cap_upload")
        if cap is not None and st.button("🔍 Lancer l'audit de réassort", type="primary"):
            dest = CAPTURES_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{slugify(Path(cap.name).stem)}{Path(cap.name).suffix.lower()}"
            dest.write_bytes(cap.getbuffer())
            with st.spinner("Analyse du rayon en cours…"):
                try:
                    st.session_state["audit"] = run_audit(
                        str(REFERENCES_DIR / ref_sel), str(dest), detector=detector, conf=conf,
                        interval=interval, use_vlm=use_vlm)
                except Exception as e:
                    st.session_state.pop("audit", None)
                    st.error(f"L'audit a échoué : {e}")

        res = st.session_state.get("audit")
        if res:
            s = res["summary"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Produits à réassortir", s["nb_anomalies"])
            m2.metric("Boîtes à sortir de la réserve", s["total_boites_a_sortir"])
            m3.metric("Images analysées", len(s["frames_analysees"]))
            for w in s["avertissements"]:
                st.warning(w)

            st.image(bgr_to_rgb(cv2.imread(res["annotated_path"])), width="stretch",
                     caption="🟥 Rupture (emplacement vide)   🟧 Colonne incomplète   "
                             "▭ gris : trou détecté entre deux boîtes")

            manifest = res["manifest"]
            if not manifest:
                st.success("✅ Rayon complet : rien à réassortir.")
            else:
                st.subheader("📋 Liste de réassort")
                df = pd.DataFrame([{
                    "Fait": False, "Produit": m["produit"], "Emplacement": m["emplacement"],
                    "Anomalie": m["type_anomalie"],
                    "En rayon": f"{m['quantite_presente']}/{m['capacite_totale']}",
                    "À sortir": m["quantite_a_reassortir"],
                } for m in manifest])
                checked = st.data_editor(
                    df, hide_index=True, width="stretch", key="checklist",
                    disabled=["Produit", "Emplacement", "Anomalie", "En rayon", "À sortir"],
                    column_config={"Fait": st.column_config.CheckboxColumn("✓ Fait")})
                done = int(checked["Fait"].sum())
                st.progress(done / len(manifest), text=f"{done}/{len(manifest)} lignes traitées")

                d1, d2, d3 = st.columns(3)
                d1.download_button("⬇️ Checklist (Excel / CSV)",
                                   checked.to_csv(index=False, sep=";").encode("utf-8-sig"),
                                   "reassort.csv", "text/csv")
                d2.download_button("🖨️ Checklist imprimable (HTML)",
                                   printable_checklist(manifest, f"Réassort — {Path(ref_sel).stem}").encode("utf-8"),
                                   "reassort_checklist.html", "text/html")
                d3.download_button("⬇️ Manifeste JSON",
                                   json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
                                   "reassort_manifest.json", "application/json")

                with st.expander("🔎 Preuves (zones vides et produit attendu)"):
                    for a in res["anomalies"]:
                        c1, c2, c3 = st.columns([1, 1, 3])
                        if a.get("crop_ref") and Path(a["crop_ref"]).exists():
                            c1.image(a["crop_ref"], caption="Produit attendu", width=110)
                        if a.get("image_preuve") and Path(a["image_preuve"]).exists():
                            c2.image(a["image_preuve"], caption="Rayon actuel", width=160)
                        c3.markdown(f"**#{a['id']} {a['produit']}**  \n"
                                    f"Étagère {a['etagere']}, Colonne {a['colonne']} — "
                                    f"{a['type_anomalie']} : {a['quantite_presente']}/{a['capacite_totale']} "
                                    f"→ **sortir {a['quantite_a_reassortir']}**")
