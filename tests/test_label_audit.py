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
