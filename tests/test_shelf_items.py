"""Espaces entre deux produits détectés d'une même tablette (données synthétiques)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shelf_items import assign_items, draw_proof, find_item_gaps, new_gaps_only  # noqa: E402


def _scene():
    # 4 étiquettes sur une rangée (y 300-330), produits posés au-dessus (y 150-290)
    slots = [{"bbox": [x, 300, x + 50, 330], "rangee": 1, "index": i + 1,
              "zone": [x - 12, 160, x + 62, 270], "etat": "occupe", "mesures": {}}
             for i, x in enumerate((20, 120, 220, 320))]
    items = [{"id": 1, "bbox": [10, 150, 90, 292]}, {"id": 2, "bbox": [95, 150, 175, 292]},
             # un produit manque entre 175 et 280
             {"id": 3, "bbox": [280, 150, 360, 292]}]
    img = np.full((400, 420, 3), 200, np.uint8)
    img[150:292, 10:175] = 90
    img[150:292, 280:360] = 90
    depth = np.full((400, 420), 10.0, np.float32)     # avant du rayon (proche)
    depth[150:292, 176:279] = 4.0                     # le mur, loin derrière, visible dans l'espace
    return img, depth, slots, items


def test_items_assigned_to_labels():
    img, depth, slots, items = _scene()
    per_row = assign_items(items, slots)
    assert len(per_row[1]) == 3
    assert slots[0]["produits"] == [1] and slots[2]["produits"] == []


def test_gap_between_items_found_once():
    img, depth, slots, items = _scene()
    gaps = find_item_gaps(img, assign_items(items, slots), slots, depth)
    assert len(gaps) == 1
    assert gaps[0]["mesures"]["entre_produits"] == [2, 3]
    assert 175 <= gaps[0]["zone"][0] and gaps[0]["zone"][2] <= 280
    # déjà signalé par le contrôle d'étiquette → pas de doublon
    assert new_gaps_only(gaps, [{"zone": gaps[0]["zone"], "etat": "vide"}]) == []
    assert draw_proof(img, items, slots, gaps).shape == img.shape


def test_no_gap_when_shelf_full():
    img, depth, slots, items = _scene()
    items.insert(2, {"id": 4, "bbox": [180, 150, 275, 292]})
    depth[150:292, 176:279] = 10.0
    assert find_item_gaps(img, assign_items(items, slots), slots, depth) == []
