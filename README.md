# Portfolio — Antoine Bui

Portfolio personnel : Data Scientist · Data Analyst · Ingénieur IA.

Site statique (HTML / CSS / JavaScript, sans dépendance), hébergé sur GitHub Pages.

## Lancer en local
Ouvrir `index.html` dans un navigateur.

## Structure
- `index.html` : contenu du site
- `style.css` : mise en forme (thème clair/sombre, responsive)
- `script.js` : bascule de thème et animations
- `assets/` : CV au format PDF
- `badminton/` : démo Badminton Vision (MediaPipe et onnxruntime-web chargés depuis un CDN)
  - volant : TrackNetV3 (licence MIT, `models/TrackNetV3-LICENSE.txt`) exécuté via WebGPU ; détecteur simple de secours sans WebGPU
  - main des joueurs (droitier/gaucher) détectée automatiquement : MediaPipe Pose au moment de chaque frappe, le poignet le plus proche du volant vote
  - `models/tracknet.onnx` (22,7 Mo) se regénère avec `tools/export_tracknet.py` (mode d'emploi en tête du script)
  - types de coups : les 18 types de ShuttleSet, prédits par un petit réseau (`models/shots.json`, 48 Ko) entraîné sur ShuttleSet22
    (CoachAI, MIT, 51 272 coups de 58 matchs pros) ; précision 80,5 % (top-2 94,3 %) en validation croisée par match.
    Regénération : `tools/build_dataset.py` puis `tools/train_shots.py` (modes d'emploi en tête des scripts)
  - arbitre virtuel (IN / OUT / FILET, score 21 points, sets) et statistiques du match, idées reprises de CoachAI
  - export CSV coup par coup au format ShuttleSet (bouton « Exporter CSV »)
- `badminton-prediction/` : copie de Badminton Vision + prédiction du prochain coup
  - ShuttleNet (CoachAI, AAAI 2022, MIT) réentraîné sans identité de joueur (serveur / receveur) sur ShuttleSet22,
    exporté en ONNX (`models/forecast.onnx`, 641 Ko, exécuté en WASM) ; prédiction dès le 4ᵉ coup d'un échange
  - sur 6 matchs non vus : bon type 48,7 % (top 3 : 84,1 %), zone de chute à 2,3 m près
    (référence « coup le plus fréquent après le précédent » : 45,1 % / 83,1 % / 3,9 m)
  - regénération : `tools/build_rallies.py`, `tools/train_forecast.py`, `tools/export_forecast.py` (mode d'emploi en tête du premier)
  - pas de vidéo fournie : l'utilisateur charge la sienne (fichier ou caméra), elle reste sur son appareil
  - tests de la logique : `node badminton/logic.test.mjs`
  - en local, la démo doit être servie en HTTP (modules ES) : `npx serve .` puis ouvrir `/badminton/`
