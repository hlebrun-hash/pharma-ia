# 💊 Pharmacie Réassort AI

Détection **100 % locale et gratuite** des ruptures et colonnes incomplètes dans un rayon
de pharmacie, à partir d'une photo ou d'une vidéo, et génération de la liste exacte des
boîtes à sortir de la réserve. Aucun compte, aucune clé d'API, aucune donnée envoyée en ligne.

## Mode recommandé : les trous au-dessus des étiquettes (sans référence)

Une étiquette de prix sur le bord d'une tablette = un emplacement produit. Il suffit donc
de filmer le rayon :

1. les **étiquettes de prix électroniques** (cadre blanc + écran gris) sont repérées ;
2. la **zone juste au-dessus** de chaque étiquette est examinée : YOLO y voit-il un
   produit ? y a-t-il des contours, du texte ou des couleurs d'emballage ? la zone est-elle
   **unie** comme le mur du fond (une boîte blanche, elle, a des ombres et des arêtes) ?
   Les mesures se font sur la zone ramenée à une taille fixe : mêmes seuils en 576 px, 1080p ou 4K.
   Si le produit attendu sort de l'image (étiquette tout en haut du cadre), l'emplacement
   n'est pas jugé.
3. rien de tout ça et zone unie → **emplacement vide** → trou à remplir, avec une photo de
   preuve (🟥 la zone vide, 🟪 l'étiquette, dont le prix aide à retrouver le produit).
4. si l'emplacement est occupé, une **carte de profondeur** (modèle local Depth Anything V2
   Small, ~100 Mo téléchargés une fois dans `models/hf/`) regarde entre l'avant du rayon
   (l'étiquette) et le mur du fond. Une partie n'est « vide » que si elle est profonde **et**
   unie (un flacon transparent semble profond, mais il porte du texte) :
   - 🟧 **espace** : on voit le fond sur au moins 45 % de la largeur → un produit manque
     à côté des autres ;
   - 🟦 **pile incomplète** : des produits sont bien à l'avant, mais on voit le mur sur au
     moins la moitié de la hauteur au-dessus d'eux → il manque des boîtes sur la pile.
   Le **retrait** de la façade (boîtes de devant vendues) est mesuré mais n'est plus signalé :
   sur les vidéos de test, il ne donnait que de fausses alertes.

5. **Nom du produit manquant** : lu sur l'**étiquette électronique** elle-même (nom court +
   prix, OCR local RapidOCR) dans l'image en **pleine résolution** — d'où l'intérêt de filmer
   en 4K ou 1080p — puis cherché dans `data/base_produits.csv` (voir plus bas).

6. *(option `--items`, très lent, expérimental)* **chaque produit visible est détecté et
   numéroté** (Grounding DINO tiny à vocabulaire ouvert, « une boîte. un flacon. un tube. »,
   sur l'image entière + 6 tuiles agrandies, fusionné avec YOLO). Une image de preuve par image
   (`preuves/detection_XX.jpg`) montre chaque produit numéroté. Ces produits servent aussi à
   trouver les **espaces entre deux produits voisins** et les trous sans étiquette (point 7).
   Sur un PC sans carte graphique : plusieurs minutes **par image** (résultat mis en cache à
   côté de l'image), et davantage de fausses alertes : désactivé par défaut.

7. *(avec `--items`)* **Sans étiquette (étiquettes papier, ou aucune) : la carte du mur.** Le mur du fond est
   repéré là où l'image est **unie** (pas de texte ni de motif) **et** nettement **plus loin**
   que la façade des boîtes voisines (profondeur interpolée entre les boîtes détectées : marche
   aussi sur une photo prise de biais). Les boîtes posées les unes sur les autres forment des
   piles, les piles alignées forment une tablette, puis :
   - 🟧 **espace** : on voit le mur entre deux piles sur au moins 0,6 largeur de boîte →
     autant de boîtes manquantes que de largeurs de boîte ;
   - 🟦 **pile incomplète** : on voit le mur au-dessus d'une pile, jusqu'à la tablette du
     dessus, sur au moins la moitié de la hauteur de l'objet du dessus (le détecteur encadre
     souvent 2 boîtes empilées comme une seule). Jamais sur la tablette la plus haute (pas de
     tablette au-dessus).
   Il faut le mur **et** l'absence de boîte : une boîte ratée par le détecteur est texturée et
   à l'avant, elle ne crée pas de faux trou. Le mur visible est teinté en vert sur les preuves.
   Limite : un objet très sombre et sans motif (flacon noir non détecté) peut passer pour du mur.

Les images trop floues de la vidéo sont écartées (elles créent de fausses étiquettes).
Option `--no-depth` : ne chercher que les emplacements vides (plus rapide).

Un même trou vu sur plusieurs images de la vidéo n'est compté qu'une fois (certitude
« forte » s'il est vu au moins deux fois) : les images sont recalées entre elles (points ORB +
homographie RANSAC) ; si les deux étiquettes ont été lues, leurs noms tranchent ; une vue qui
ressemble à deux trous déjà vus les fusionne.

```powershell
python src/pipeline.py --input "video teste .mp4"
python src/pipeline.py --input "video teste .mp4" --items   # + chaque produit (très lent)
```

Dans l'interface (`streamlit run app.py`) : une seule page, on envoie la photo ou la vidéo du
rayon, rien d'autre. **Aucune photo de référence** : seules les images envoyées sont analysées.

**Nom du produit (OCR local, RapidOCR)** : pour un emplacement à réassortir, le nom vient
d'abord de l'**étiquette électronique** (nom court, ex. « SEBIUM KERATO+ », et prix), lue dans
l'image d'origine en pleine résolution ; avec `--items`, aussi du texte des produits restants
sur l'emplacement. Le texte est ensuite cherché (recherche approximative, tolère les fautes de
lecture) dans `data/base_produits.csv` — une ligne par produit, colonne `nom` obligatoire,
toute autre colonne (code CIP, référence, fournisseur…) est reprise dans le résultat. Sans ce
fichier, la recherche se fait dans les noms des dossiers de `data/catalogue/`. Option `--no-ocr`.
Une étiquette de moins de 60 px de large dans l'image n'est pas lisible : le nom reste alors
« à identifier (voir l'étiquette sur la photo) ».

### Pour filmer (important pour la fiabilité)

* **4K ou 1080p** (réglage de l'appareil photo du téléphone) : en 576 px, le nom écrit sur
  les étiquettes est illisible ;
* **de face**, à **environ 1 m**, en **travelling lent** (une tablette après l'autre si le
  rayon est haut), étiquettes bien visibles, sans contre-jour ;
* chaque emplacement doit apparaître sur **au moins 2 images** (≈ 3 s à l'écran) : un trou vu
  deux fois est signalé avec une certitude « forte ».

### Mesurer la fiabilité : `scripts/evaluer.py`

Le taux de bons résultats se **mesure** sur des vidéos dont on connaît les vrais trous
(vérité terrain dans `data/verite_terrain/<video>.json`, format décrit en tête du script) :

```powershell
python scripts/evaluer.py                 # toutes les vidéos annotées
python scripts/evaluer.py video_teste     # une seule
```

Chaque alerte est classée JUSTE / DOUBLON / FAUSSE (ou IGNORÉE si elle tombe sur un cas
ambigu annoté) ; le script affiche la **précision** (alertes justes / alertes) et le **rappel**
(vrais trous trouvés / vrais trous). Pour obtenir une mesure solide : retirer volontairement
quelques produits d'un rayon (noter lesquels), filmer, annoter les trous, recommencer sur
plusieurs rayons — au moins 20 à 30 vrais trous au total.

Limites actuelles : une boîte blanche, vue parfaitement de face, sans texte ni ombre, peut
encore passer pour le mur ; un détecteur d'étiquettes qui rate une étiquette ne contrôle pas
l'emplacement au-dessus (trou non vu).

## Ancien mode « référence » (ligne de commande seulement, retiré de l'interface)

> Ce mode demande une photo du rayon plein ; il n'est plus proposé dans l'interface et n'est
> pas utilisé par le mode « étiquettes » ci-dessus. Il reste accessible avec `--ref`.

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
streamlit run app.py      # puis envoyer une photo ou une vidéo du rayon
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
    mode étiquettes : [--no-depth] [--no-ocr] [--items]
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

### Identification des produits (ancien mode « référence » uniquement)
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
3. **Saisie manuelle** dans `data/references/<rayon>.planogram.json`.

## Conseils de prise de vue

Voir « Pour filmer » plus haut : 4K ou 1080p, de face, à environ 1 m, travelling lent,
sans contre-jour.

## Structure

```
src/video_engine.py       frames nettes (variance du Laplacien)
src/shelf_detector.py     boîtes (YOLO / OpenCV), rangées, étagères, trous
src/planogram_matcher.py  planogramme, alignement, appariement hongrois, Δ par bloc
src/local_vlm.py          Ollama (Qwen2.5-VL) + catalogue FAISS
src/label_detector.py     étiquettes de prix électroniques (OpenCV)
src/slot_checker.py       zone au-dessus de chaque étiquette : vide / espace / incomplet / occupée
src/depth_estimator.py    profondeur monoculaire locale (Depth Anything V2 Small)
src/item_detector.py      chaque produit visible (Grounding DINO tiny + YOLO), preuve numérotée
src/shelf_items.py        produits → tablettes/étiquettes, espaces entre produits voisins
src/wall_map.py           carte du mur (profondeur + surface unie), trous sans étiquette
src/text_reader.py        texte des boîtes et étiquettes (RapidOCR) → produit de la base
src/label_audit.py        mode « étiquettes » : trous d'une photo/vidéo, sans référence
src/pipeline.py           orchestrateur CLI
app.py                    interface Streamlit
scripts/generate_demo_data.py   jeu de démonstration synthétique
scripts/evaluer.py        précision / rappel sur les vidéos annotées (data/verite_terrain/)
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
