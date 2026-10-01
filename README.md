# Détection de fatigue du conducteur

Programme Python qui surveille le visage d'une personne à la caméra et déclenche une alerte quand il détecte des signes de fatigue. Projet d'étude, École supérieure de technologie (Salé), 2024-2025.

## Ce qu'il détecte

- **Yeux fermés et clignements** : rapport d'ouverture des yeux (EAR) sous un seuil (0,25), avec un compteur de clignements par minute.
- **Bâillements** : rapport d'ouverture de la bouche (MAR) au-dessus d'un seuil (0,30) pendant un certain nombre d'images consécutives.
- **Main trop proche du visage** : alerte selon la distance en pixels.

Une interface Tkinter affiche l'image et l'état, avec une alerte sonore et vocale.

## Deux versions

- `src/detection_mediapipe.py` : MediaPipe Face Mesh (version principale).
- `src/detection_dlib.py` : dlib, avec les 68 points du visage. Il faut télécharger séparément `shape_predictor_68_face_landmarks.dat` et le placer à côté du script.

## Installation

```bash
pip install -r requirements.txt
python src/detection_mediapipe.py
```

Ce code utilise `winsound`, il fonctionne donc sous Windows.
