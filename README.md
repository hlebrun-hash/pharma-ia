# 💊 Pharmacie Réassort AI

Détection **100 % locale et gratuite** des ruptures et colonnes incomplètes dans un rayon
de pharmacie, à partir d'une photo ou d'une vidéo, et génération de la liste exacte des
boîtes à sortir de la réserve. Aucun compte, aucune clé d'API, aucune donnée envoyée en ligne.

## Mode recommandé : les trous au-dessus des étiquettes (sans référence)

Une étiquette de prix sur le bord d'une tablette = un emplacement produit. Il suffit donc
de filmer le rayon :

1. les **étiquettes de prix électroniques** (cadre blanc + écran gris) sont repérées ;
2. la **zone juste au-dessus** de chaque étiquette est examinée : YOLO y voit-il un
   produit ? y a-t-il des contours, du texte ou des couleurs d'emballage ?
3. rien de tout ça → **emplacement vide** → trou à remplir, avec une photo de preuve
   (🟥 la zone vide, 🟪 l'étiquette, dont le prix aide à retrouver le produit).
4. si l'emplacement est occupé, une **carte de profondeur** (modèle local Depth Anything V2
   Small, ~100 Mo téléchargés une fois dans `models/hf/`) regarde entre l'avant du rayon
   (l'étiquette) et le mur du fond :
   - 🟧 **espace** : on voit le fond sur au moins 45 % de la largeur → un produit manque
     à côté des autres ;
   - 🟦 **pile incomplète** : des produits sont bien à l'avant, mais on voit le mur sur au
     moins la moitié de la hauteur au-dessus d'eux → il manque des boîtes sur la pile ;
   - 🟨 **retrait** : toute la façade est loin derrière l'étiquette → les boîtes de devant
     ont été vendues, il ne reste que celles du fond.

5. **chaque produit visible est détecté et numéroté** (Grounding DINO tiny à vocabulaire
   ouvert, « une boîte. un flacon. un tube. », sur l'image entière + 6 tuiles agrandies, fusionné
   avec YOLO). Une image de preuve par image analysée (`preuves/detection_XX.jpg`) montre
   chaque produit numéroté et, sous chaque étiquette, le nombre de produits vus au-dessus.
   Ces produits servent aussi à trouver les **espaces entre deux produits voisins**.
   Lent : ~1,5 min par image la première fois (résultat mis en cache à côté de l'image) ;
   option `--no-items` pour s'en passer. Les modèles COCO (YOLO, RT-DETR) ne connaissent pas
   la classe « boîte » : testés, ils ne trouvent que les flacons.

Les images trop floues de la vidéo sont écartées (elles créent de fausses étiquettes).
Option `--no-depth` : ne chercher que les emplacements vides (plus rapide).

Un même trou vu sur plusieurs images de la vidéo n'est compté qu'une fois (certitude
« forte » s'il est vu au moins deux fois).

```powershell
python src/pipeline.py --input "video teste .mp4"
```

Dans l'interface : onglet **① Trous dans le rayon (étiquettes)**.

**Nom du produit (OCR local, RapidOCR)** : le texte écrit sur chaque produit détecté est lu
(nom affiché sur l'image de preuve). Pour un emplacement à réassortir, le nom vient des
produits restants sur l'emplacement (même pile = même produit), sinon du texte de l'étiquette.
Ce texte est ensuite cherché (recherche approximative, tolère les fautes de lecture) dans
`data/base_produits.csv` — une ligne par produit, colonne `nom` obligatoire, toute autre
colonne (code CIP, référence, fournisseur…) est reprise dans le résultat. Sans ce fichier, la
recherche se fait dans les noms des dossiers de `data/catalogue/`. Option `--no-ocr`.

Limites actuelles : un emplacement entièrement vide n'a plus de boîte à lire ; son nom dépend
alors de l'étiquette, souvent trop petite en vidéo (on indique alors les produits voisins) ;
une boîte entièrement blanche et sans motif peut être prise pour un trou.
Pour filmer : de face, à environ 1 m, lentement, étiquettes bien visibles.

## Mode « référence » : comparaison avec une photo du rayon plein

1. **Rayon plein (une seule fois)** : on photographie le rayon bien rangé. Chaque boîte est
   détectée, puis les boîtes voisines identiques sont regroupées en *blocs produit*
   (ex. « Doliprane 1000 mg : 4 colonnes × 2 rangs = 8 boîtes »). Les noms des produits
   peuvent être lus automatiquement (VLM local / catalogue) ou saisis à la main.
2. **Audit du jour** : photo ou vidéo du même rayon. La capture est recalée sur la
   référence (homographie SIFT + RANSAC), chaque emplacement est apparié par
   l'algorithme hongrois et son apparence est comparée à la référence.
3. **Résultat** : pour chaque bloc, `Δ = capacité − quantité présente` → manifeste JSON,
   image annotée (🟥 rupture, 🟧 colonne incomplète) et checklist imprimable.

> Les quantités sont comptées **en façade** (boîtes visibles de face). Si un emplacement
> contient plusieurs boîtes en profondeur, une boîte manquante n'est vue que lorsque la
> façade est vide.

## Installation (Windows, PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
# Sans carte graphique NVIDIA, la version CPU de torch est bien plus légère :
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Démo en 30 secondes

```powershell
python scripts/generate_demo_data.py
python src/pipeline.py --ref data/references/rayon_01_plein.jpg --input data/captures/visite_rayon_01.mp4
streamlit run app.py
```

Sorties dans `data/outputs/` :

| Fichier | Contenu |
|---|---|
| `reassort_manifest.json` | liste à réassortir (format ci-dessous) |
| `audit_annote.jpg` | image avec les zones à remplir |
| `audit_details.json` | diagnostic complet (alignement, coordonnées, trous détectés) |

```json
[
  {
    "id": 1,
    "produit": "Doliprane 1000mg Comprimés",
    "emplacement": "Étagère 1, Colonne 1",
    "type_anomalie": "Colonne incomplète",
    "quantite_presente": 5,
    "capacite_totale": 8,
    "quantite_a_reassortir": 3,
    "image_preuve": "data/crops/trou_et1_col1.jpg"
  }
]
```

## Options du pipeline

```
python src/pipeline.py [--ref <photo_rayon_plein>] --input <photo_ou_video>
    (sans --ref : mode « étiquettes »)
    [--detector auto|classical|yolo] [--weights models/xxx.pt] [--conf 0.25]
    [--interval 1.5] [--no-vlm] [--rebuild-planogram] [--out-dir data/outputs]
```

## Modèles (tous optionnels, tous locaux)

### Détection des boîtes
* **`classical`** (par défaut s'il n'y a pas de poids) : segmentation OpenCV, aucun
  téléchargement. Convient à un rayon propre, bien éclairé, photographié de face.
* **`yolo`** : pour des photos réelles chargées (boîtes serrées, reflets), déposez dans
  `models/` un modèle de détection de produits en rayon entraîné sur **SKU-110K** dont le
  nom contient `sku`, `shelf`, `retail`, `rayon` ou `pharma` (ex. `models/sku110k_yolov8.pt`)
  ; il sera utilisé automatiquement en mode `auto`. On peut aussi entraîner le sien avec
  Ultralytics sur le jeu public SKU-110K (`yolo detect train data=SKU-110K.yaml model=yolov8m.pt`).
  Le modèle générique COCO `yolov8m.pt` ne connaît pas les « boîtes de médicament » et
  n'est utile qu'en dernier recours (`--detector yolo --weights models/yolov8m.pt`).

> ⚠️ Les fichiers `.pt` sont des *pickles* Python : ne chargez que des poids de source sûre.

### Identification des produits
1. **VLM local Ollama** (recommandé si la machine a ≥ 16 Go de RAM) :
   ```powershell
   winget install Ollama.Ollama
   ollama pull qwen2.5vl:7b
   ```
   (le tag officiel est `qwen2.5vl`, sans tiret). Modèle plus léger pour PC modeste :
   `ollama pull qwen2.5vl:3b` puis `setx PHARMA_VLM_MODEL qwen2.5vl:3b`.
2. **Catalogue visuel** (fallback, très léger) : une photo de face par produit dans
   `data/catalogue/<Nom du produit>/photo.jpg`. Indexation FAISS d'embeddings
   MobileNetV3 (poids ImageNet téléchargés une fois depuis pytorch.org, puis hors-ligne).
3. **Saisie manuelle** dans l'onglet ① de l'interface : les noms sont mémorisés dans
   `data/references/<rayon>.planogram.json` et réutilisés à chaque audit.

## Conseils de prise de vue

* Même position et même cadrage que la photo de référence (au mieux : repère au sol).
* Rayon entier dans le champ, de face, sans contre-jour ; en vidéo, travelling lent.
* Refaire la photo de référence quand l'implantation (planogramme) change.

## Structure

```
src/video_engine.py       frames nettes (variance du Laplacien)
src/shelf_detector.py     boîtes (YOLO / OpenCV), rangées, étagères, trous
src/planogram_matcher.py  planogramme, alignement, appariement hongrois, Δ par bloc
src/local_vlm.py          Ollama (Qwen2.5-VL) + catalogue FAISS
src/label_detector.py     étiquettes de prix électroniques (OpenCV)
src/slot_checker.py       zone au-dessus de chaque étiquette : vide / espace / retrait / occupée
src/depth_estimator.py    profondeur monoculaire locale (Depth Anything V2 Small)
src/item_detector.py      chaque produit visible (Grounding DINO tiny + YOLO), preuve numérotée
src/shelf_items.py        produits → tablettes/étiquettes, espaces entre produits voisins
src/text_reader.py        texte des boîtes et étiquettes (RapidOCR) → produit de la base
src/label_audit.py        mode « étiquettes » : trous d'une photo/vidéo, sans référence
src/pipeline.py           orchestrateur CLI
app.py                    interface Streamlit
scripts/generate_demo_data.py   jeu de démonstration synthétique
tests/test_e2e.py         test end-to-end (photo + vidéo)
tests/test_label_audit.py mode étiquettes (vrai trou, trou simulé, manifeste)
repos_externes/           dépôts open-source de référence (non importés à l'exécution)
```

### Code réutilisé
* [Shelf-Vision](https://github.com/findsamirks-commits/Shelf-Vision) (MIT) : coût
  d'appariement hongrois `(1 − IoU) + 0.5 × distance des centres` → `planogram_matcher.py`.
* [Retail-Shelf-Monitoring](https://github.com/Alijanloo/Retail-Shelf-Monitoring) (MIT) :
  recalage par homographie RANSAC et validation de la matrice, catalogue d'embeddings
  FAISS, consensus multi-images → `planogram_matcher.py`, `local_vlm.py`.

Les poids `weights/best.pt` de Shelf-Vision ne sont **pas** utilisés : ils sont entraînés
sur 12 produits d'épicerie synthétiques et ne détectent pas des boîtes de médicaments.

## Tests

```powershell
python -m pytest tests -q
```
