"""Lecture du texte (OCR) et recherche du produit dans la base."""

import cv2
import numpy as np
import pytest

from src import text_reader as tr


@pytest.fixture
def base(monkeypatch):
    rows = ({"nom": "DOLIPRANE 1000mg", "cip": "3400930000001"},
            {"nom": "SPASFON LYOC 80mg", "cip": "3400930000002"},
            {"nom": "GAVISCON 24 sach", "cip": "3400930000003"})
    monkeypatch.setattr(tr, "load_base", lambda: rows)
    return rows


def test_match_brand_read_on_box(base):
    m = tr.match_base(["DOLIPRANE", "paracétamol"])
    assert m and m["nom"] == "DOLIPRANE 1000mg" and m["cip"] == "3400930000001"


def test_match_tolerates_ocr_typos(base):
    m = tr.match_base(["SPASF0N LY0C 80 mg"])
    assert m and m["nom"] == "SPASFON LYOC 80mg"


def test_unknown_text_not_matched(base):
    assert tr.match_base(["VICHY MINERAL 89"]) is None
    ident = tr.identify(["VICHY MINERAL 89"])
    assert ident["source"] == "texte lu" and ident["nom"] == "VICHY MINERAL 89"


def test_main_name_is_biggest_text():
    lines = [{"texte": "20 comprimés", "conf": 0.9, "hauteur": 12},
             {"texte": "DOLIPRANE", "conf": 0.95, "hauteur": 40},
             {"texte": "paracétamol", "conf": 0.9, "hauteur": 15}]
    assert tr.main_name(lines) == "DOLIPRANE"


def test_lines_assigned_to_the_box_that_contains_them():
    lines = [{"texte": "EUPHYTOSE", "conf": 0.9, "hauteur": 30, "box": [110, 20, 190, 50]},
             {"texte": "DEXERYL", "conf": 0.9, "hauteur": 30, "box": [310, 20, 390, 50]}]
    items = [{"id": 1, "bbox": [100, 0, 200, 200]}, {"id": 2, "bbox": [300, 0, 400, 200]},
             {"id": 3, "bbox": [500, 0, 600, 200]}]
    tr.read_items(None, items, lines)
    assert [it["nom_lu"] for it in items] == ["EUPHYTOSE", "DEXERYL", ""]


def test_ocr_reads_synthetic_box():
    pytest.importorskip("rapidocr_onnxruntime")
    img = np.full((200, 420, 3), 255, np.uint8)
    cv2.putText(img, "DOLIPRANE", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    lines = tr.read_text(img)
    assert any("DOLIPRANE" in l["texte"].upper() for l in lines)
