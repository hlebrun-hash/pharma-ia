"""Pharmacie Réassort AI — pipeline 100 % local de détection des ruptures en rayon."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REFERENCES_DIR = DATA_DIR / "references"
CAPTURES_DIR = DATA_DIR / "captures"
CROPS_DIR = DATA_DIR / "crops"
OUTPUTS_DIR = DATA_DIR / "outputs"
CATALOGUE_DIR = DATA_DIR / "catalogue"
MODELS_DIR = ROOT_DIR / "models"

for _d in (REFERENCES_DIR, CAPTURES_DIR, CROPS_DIR, OUTPUTS_DIR, CATALOGUE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
