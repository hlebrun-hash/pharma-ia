"""Carte du mur et trous trouvés sans étiquette (scène synthétique).

Deux tablettes. Tablette du haut : pleine. Tablette du bas (bas des boîtes y=420, planche
de la tablette du dessus y=200-230) :
  pile 1 : jusqu'en haut · pile 2 : basse (mur visible au-dessus) · trou de 2 boîtes ·
  piles 3 et 4 : jusqu'en haut.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wall_map import (build_shelves, build_stacks, build_wall_map,  # noqa: E402
                          find_wall_holes)

NEAR, FAR = 10.0, 6.0          # disparité : grand = proche


def _box(img, depth, b, seed):
    x1, y1, x2, y2 = b
    rng = np.random.default_rng(seed)
    img[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)  # emballage
    depth[y1:y2, x1:x2] = NEAR


def _scene():
    img = np.full((460, 420, 3), 185, np.uint8)        # mur uni…
    depth = np.full((460, 420), FAR, np.float32)       # … loin derrière
    img[200:232] = 120                                  # planche de la tablette du dessus :
    depth[200:232] = NEAR                               # unie mais à l'avant → pas du mur
    img[420:] = 120
    depth[420:] = NEAR
    items, k = [], 0
    for x in (20, 85, 150, 215, 275, 340):              # tablette du haut, pleine
        k += 1
        b = (x, 90, x + 60, 200)
        _box(img, depth, b, k)
        items.append({"id": k, "bbox": list(map(float, b))})
    for x, top in ((20, 234), (85, 300), (275, 234), (340, 234)):   # tablette du bas
        k += 1
        b = (x, top, x + 60, 420)
        _box(img, depth, b, k)
        items.append({"id": k, "bbox": list(map(float, b))})
    return img, depth, items


def test_wall_map_sees_the_wall_not_the_boxes_nor_the_shelf():
    img, depth, items = _scene()
    wm = build_wall_map(img, depth, items)
    wall = wm["mur"]
    assert wall[330, 210]            # le trou entre les piles : on voit le mur
    assert wall[265, 115]            # au-dessus de la pile basse
    assert not wall[360, 50]         # dans une boîte
    assert not wall[215, 210]        # la planche de la tablette (unie mais à l'avant)


def test_stacks_and_shelves():
    _, _, items = _scene()
    # une 2ᵉ boîte de même largeur posée sur la pile 2 → même pile
    items.append({"id": 99, "bbox": [85.0, 240.0, 145.0, 300.0]})
    stacks = build_stacks(items)
    pile2 = [s for s in stacks if any(i["id"] == 99 for i in s["items"])][0]
    assert sorted(i["id"] for i in pile2["items"]) == [8, 99]
    shelves = build_shelves(stacks)
    assert len(shelves) == 2                       # la boîte du haut n'est pas « empilée »
    assert len(shelves[0]) == 6 and len(shelves[1]) == 4


def test_gap_and_low_stack_found_without_labels():
    img, depth, items = _scene()
    holes, wm = find_wall_holes(img, depth, items)
    by_state = {}
    for h in holes:
        by_state.setdefault(h["etat"], []).append(h)
    assert set(by_state) == {"espace", "incomplet"}, holes
    (gap,) = by_state["espace"]
    assert gap["manquants"] == 2                   # trou de 2 largeurs de boîte
    assert gap["mesures"]["entre_produits"] == [8, 9]
    assert gap["rangee"] == 2 and gap["source"] == "mur"
    (low,) = by_state["incomplet"]
    assert low["produits"] == [8]                  # la pile basse (le produit à remettre)
    assert low["zone"][3] == 300.0 and low["zone"][1] < 270
    # la tablette du haut n'a pas de tablette au-dessus : jamais « incomplète »
    assert all(h["rangee"] == 2 for h in holes)


def test_missed_box_is_not_a_false_hole():
    """Une boîte que le détecteur n'a pas vue est texturée et à l'avant : pas un mur."""
    img, depth, items = _scene()
    _box(img, depth, (150, 234, 210, 420), 50)
    _box(img, depth, (212, 234, 272, 420), 51)
    holes, _ = find_wall_holes(img, depth, items)
    assert [h for h in holes if h["etat"] == "espace"] == []


def test_needs_depth():
    img, _, items = _scene()
    assert find_wall_holes(img, None, items) == ([], None)
