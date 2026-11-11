"""Mode « étiquettes » : trouver un trou au-dessus d'une étiquette de prix, sans référence.

Utilise une image de la vidéo réelle si elle a été extraite (data/captures/frames_video_teste/),
sinon le test est ignoré.
"""

import json
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.label_audit import run_label_audit  # noqa: E402
from src.pipeline import MANIFEST_SCHEMA  # noqa: E402
from src.slot_checker import check_frame  # noqa: E402

FRAME_WITH_HOLE = ROOT / "data/captures/frames_video_teste/video_teste_f000353.jpg"
FRAME_FULL = ROOT / "data/captures/frames_video_teste/video_teste_f000088.jpg"

needs_frames = pytest.mark.skipif(not FRAME_WITH_HOLE.exists() or not FRAME_FULL.exists(),
                                  reason="images de la vidéo test absentes")


@needs_frames
def test_real_hole_found_and_full_shelf_quiet():
    slots = check_frame(cv2.imread(str(FRAME_WITH_HOLE)))
    assert len(slots) >= 7  # image prise de près : grandes étiquettes, moins nombreuses
    assert any(s["etat"] == "vide" for s in slots)
    full = check_frame(cv2.imread(str(FRAME_FULL)))
    assert len(full) >= 10
    assert sum(s["etat"] == "vide" for s in full) <= 1


@needs_frames
def test_simulated_hole_detected():
    img = cv2.imread(str(FRAME_FULL))
    slots = [s for s in check_frame(img) if s["etat"] == "occupe"]
    target = max(slots, key=lambda s: s["zone"][3] - s["zone"][1])
    x1, y1, x2, y2 = (int(v) for v in target["zone"])
    img[y1:y2, x1:x2] = (95, 100, 95)  # fond uni : le produit a été retiré
    after = check_frame(img)
    near = [s for s in after if abs(s["bbox"][0] - target["bbox"][0]) < 10
            and abs(s["bbox"][1] - target["bbox"][1]) < 10]
    assert near and near[0]["etat"] == "vide"


@needs_frames
def test_run_label_audit_writes_valid_manifest(tmp_path):
    r = run_label_audit(str(FRAME_WITH_HOLE), detector="classical", use_depth=False, out_dir=tmp_path)
    manifest = json.loads(Path(r["manifest_path"]).read_text(encoding="utf-8"))
    MANIFEST_SCHEMA.validate_python(manifest)
    assert manifest and manifest[0]["type_anomalie"] == "EMPLACEMENT_VIDE"
    assert (ROOT / manifest[0]["image_preuve"]).exists() or Path(manifest[0]["image_preuve"]).exists()
    assert Path(r["annotated_path"]).exists()


FRAME_WITH_SPACE = ROOT / "data/captures/frames_video_teste/video_teste_f000294.jpg"
needs_depth = pytest.mark.skipif(
    not FRAME_WITH_SPACE.exists() or not (ROOT / "models/hf").exists(),
    reason="image test ou modèle de profondeur (models/hf) absent")


@needs_depth
def test_depth_finds_space_between_products():
    from src.depth_estimator import estimate_depth

    img = cv2.imread(str(FRAME_WITH_SPACE))
    slots = check_frame(img, None, estimate_depth(img))
    # au moins un manque vu grâce à la profondeur (espace ou pile incomplète)
    assert any(s["etat"] in ("espace", "incomplet") for s in slots)
    flagged = sum(s["etat"] in ("espace", "incomplet", "retrait") for s in slots)
    assert flagged <= 3, "trop d'alertes sur un rayon presque plein"


@needs_depth
def test_depth_finds_incomplete_stack():
    """Étiquette à 6 € (0:09) : 2 petites boîtes à l'avant, mur visible au-dessus."""
    from src.depth_estimator import estimate_depth

    img = cv2.imread(str(FRAME_WITH_SPACE))
    slots = check_frame(img, None, estimate_depth(img))
    target = [s for s in slots if abs(s["bbox"][0] - 109) < 12 and abs(s["bbox"][1] - 602) < 12]
    assert target and target[0]["etat"] == "incomplet"


FRAME_CLOSE = ROOT / "data/captures/frames_video_teste/video_teste_f000441.jpg"


@pytest.mark.skipif(not FRAME_CLOSE.exists(), reason="image test absente")
def test_big_close_labels_found():
    """Vue de près : les 3 grandes étiquettes 11,00 / 6,50 / 9,99 (rangée inclinée)."""
    from src.label_detector import detect_labels

    labels = detect_labels(cv2.imread(str(FRAME_CLOSE)))
    big = [l for l in labels if l["bbox"][1] < 260 and l["bbox"][2] - l["bbox"][0] > 45]
    assert len(big) == 3


FRAME_1080 = ROOT / "data/captures/frames_IMG_0296/IMG_0296_f000172.jpg"


@pytest.mark.skipif(not FRAME_1080.exists(), reason="image test absente")
def test_flat_labels_of_second_rail_found():
    """IMG_0296 (1080p) : rangée du bas = 4 étiquettes 117 × 84 px puis, sur un autre rail,
    4 étiquettes plus « aplaties » (130 × 65 px) qui étaient ratées. Aucune fausse étiquette
    dans l'espace entre deux étiquettes voisines."""
    from src.label_detector import detect_labels

    labels = detect_labels(cv2.imread(str(FRAME_1080)))
    bottom = sorted(l["bbox"][0] for l in labels if l["bbox"][1] > 700)
    assert len(bottom) == 8
    assert all(b - a > 100 for a, b in zip(bottom, bottom[1:]))   # pas de chevauchement


@needs_frames
def test_no_label_on_products():
    """Les fausses « étiquettes » sur le bas des tubes sont retirées grâce aux produits détectés."""
    import json as _json

    from src.label_detector import detect_labels, filter_labels_by_items

    cache = Path(str(FRAME_FULL) + ".produits.json")
    if not cache.exists():
        pytest.skip("produits non détectés pour cette image")
    items = _json.loads(cache.read_text(encoding="utf-8"))["items"]
    labels = filter_labels_by_items(detect_labels(cv2.imread(str(FRAME_FULL))), items)
    for l in labels:  # aucune étiquette entre y=600 et 680 (zone des bas de tubes)
        assert not (600 < (l["bbox"][1] + l["bbox"][3]) / 2 < 680)


def test_row_where_every_slot_is_empty_is_dropped():
    """Rainures d'un mur blanc prises pour des étiquettes : toute la rangée « vide » → écartée."""
    from src.label_audit import _drop_all_empty_rows

    wall = [{"rangee": 1, "index": i, "etat": "vide"} for i in range(1, 6)]
    shelf = [{"rangee": 2, "index": 1, "etat": "vide"}, {"rangee": 2, "index": 2, "etat": "occupe"},
             {"rangee": 2, "index": 3, "etat": "occupe"}]
    kept, n = _drop_all_empty_rows(wall + shelf)
    assert n == 1 and len(kept) == 3
    assert {s["rangee"] for s in kept} == {1}          # renumérotée
    assert [s["etat"] for s in kept].count("vide") == 1  # le vrai trou reste signalé


def test_white_box_is_not_an_empty_slot():
    """Peu de contours mais luminosité contrastée (boîte blanche, ombre entre deux boîtes) :
    ce n'est pas le mur du fond → pas « vide ». Zone unie → « vide »."""
    from src.slot_checker import classify

    base = {"valide": True, "contours": 0.02, "couleur": 0.0, "yolo": False}
    assert classify({**base, "ecart_type": 20.0}) == "occupe"
    assert classify({**base, "ecart_type": 10.0}) == "vide"


def test_recess_alone_is_not_reported():
    """La façade en retrait donnait surtout de fausses alertes : elle n'est plus signalée."""
    from src.slot_checker import classify

    f = {"valide": True, "contours": 0.15, "couleur": 0.1, "ecart_type": 40.0, "yolo": False,
         "recul": 0.6, "espace": 0.0, "hauteur_vide": 0.0}
    assert classify(f) == "occupe"


def test_edge_measure_does_not_depend_on_resolution():
    """Même scène en 576 px et en 4× plus grand : mêmes mesures (seuils communs)."""
    from src.slot_checker import zone_features

    img = cv2.imread(str(FRAME_FULL)) if FRAME_FULL.exists() else None
    if img is None:
        pytest.skip("image test absente")
    zone = [200.0, 300.0, 260.0, 420.0]
    small = zone_features(img, zone, None)
    big = zone_features(cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                        [v * 4 for v in zone], None)
    assert abs(small["contours"] - big["contours"]) < 0.03
    assert abs(small["ecart_type"] - big["ecart_type"]) < 3


def test_zone_outside_image_is_not_judged():
    """Étiquette tout en haut de l'image : le produit au-dessus n'est pas visible → non évalué."""
    import numpy as np

    from src.slot_checker import check_frame

    img = np.full((400, 600, 3), 200, np.uint8)
    labels = [{"bbox": [100.0 + 120 * k, 20.0, 150.0 + 120 * k, 50.0], "rangee": 1, "index": k + 1}
              for k in range(3)]
    for s in check_frame(img, None, None, labels):
        assert s["etat"] == "non_evalue"


def test_hole_groups_linked_by_a_later_view_are_merged():
    """Trou vu sur l'image 2 (groupe A) et 3 (groupe B, non relié à A), puis sur l'image 4
    qui ressemble aux deux : un seul trou à la fin."""
    from src.label_audit import _merge_groups

    def grp(img, fi):
        return {"vues": [{"image": img, "etat": "vide", "zone": [0, 0, 1, 1]}], "last_frame": fi,
                "sig": None, "last_zone": [0, 0, 1, 1],
                "slot": {"etat": "vide", "identification": {"nom": "BODY SPRAY" if fi == 3 else ""}}}

    a, b = grp("f002.jpg", 2), grp("f003.jpg", 3)
    _merge_groups(a, b)
    assert [v["image"] for v in a["vues"]] == ["f002.jpg", "f003.jpg"]
    assert a["last_frame"] == 3
    assert a["slot"]["identification"]["nom"] == "BODY SPRAY"   # nom lu sur l'autre vue gardé


def test_parse_electronic_label():
    from src.text_reader import parse_label

    r = parse_label(["SEBIUM KERATO+", "GEL CREME", "1890", "40ML", "3401399373893"])
    assert r["nom"] == "SEBIUM KERATO+ GEL CREME"
    assert r["prix"] == "18,90 €"
    assert parse_label(["13€50"])["prix"] == "13,50 €"


def test_evaluation_scoring():
    """scripts/evaluer.py : juste / doublon / ignorée / fausse."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluer import score

    gt = {"trous": [{"id": "t1", "vues": {"a.jpg": [100, 100, 200, 200]}}],
          "ignorer": {"b.jpg": [[0, 0, 50, 50]]}}

    def alert(i, img, zone):
        return {"id": i, "etat": "vide", "produit": "", "zone": zone,
                "vu_sur_images": [{"image": img, "zone": zone}]}

    r = score(gt, [alert(1, "a.jpg", [110, 110, 190, 190]), alert(2, "a.jpg", [120, 120, 180, 180]),
                   alert(3, "b.jpg", [5, 5, 40, 40]), alert(4, "b.jpg", [300, 300, 350, 350])], {})
    assert [a["statut"] for a in r["alertes"]] == ["JUSTE", "DOUBLON", "IGNORÉE", "FAUSSE"]
    assert r["trouves"] == 1 and r["vrais_trous"] == 1
