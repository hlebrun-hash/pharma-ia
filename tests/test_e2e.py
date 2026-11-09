"""Test end-to-end sur données synthétiques.

    python -m pytest tests -q        (ou)        python tests/test_e2e.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import MANIFEST_SCHEMA  # noqa: E402

REF = ROOT / "data/references/rayon_01_plein.jpg"
PHOTO = ROOT / "data/captures/rayon_01_actuel.jpg"
VIDEO = ROOT / "data/captures/visite_rayon_01.mp4"
TRUTH = ROOT / "data/demo_verite_terrain.json"
MANIFEST = ROOT / "data/outputs/reassort_manifest.json"


def _generate():
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_demo_data.py")], check=True, cwd=ROOT)


def _run(inp: Path) -> list[dict]:
    subprocess.run([sys.executable, str(ROOT / "src/pipeline.py"), "--ref", str(REF), "--input", str(inp),
                    "--no-vlm", "--detector", "classical", "--rebuild-planogram"], check=True, cwd=ROOT)
    assert MANIFEST.exists()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    MANIFEST_SCHEMA.validate_python(data)
    assert data, "Le manifeste ne doit pas être vide"
    return data


def _expected() -> set[tuple[str, int, int]]:
    t = json.loads(TRUTH.read_text(encoding="utf-8"))
    return {(f"Étagère {a['etagere']}, Colonne {a['colonne']}", a["capacite"] - a["manquants"], a["capacite"])
            for a in t["anomalies_attendues"]}


def _got(manifest: list[dict]) -> set[tuple[str, int, int]]:
    return {(m["emplacement"], m["quantite_presente"], m["capacite_totale"]) for m in manifest}


def test_photo_end_to_end():
    _generate()
    assert _got(_run(PHOTO)) == _expected()


def test_video_end_to_end():
    if not VIDEO.exists():
        _generate()
    assert _got(_run(VIDEO)) == _expected()


if __name__ == "__main__":
    test_photo_end_to_end()
    test_video_end_to_end()
    print("OK : tous les tests end-to-end passent.")
