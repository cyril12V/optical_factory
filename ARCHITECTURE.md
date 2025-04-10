# Architecture Système - Optical Factory

Ce document décrit l'architecture système actuelle du projet Optical Factory, axée sur l'utilisation d'un modèle MLP pour l'analyse faciale et des tests d'évaluation.

## 1. Vue d'Ensemble

L'application permet aux utilisateurs d'obtenir une classification de la forme de leur visage via leur webcam et de recevoir des recommandations de lunettes associées.

*   **Frontend (React)** : Capture le flux vidéo, utilise **TensorFlow.js (Facemesh)** pour extraire 468 landmarks faciaux, et appelle le backend pour l'analyse.
*   **Backend (FastAPI)** : Reçoit les landmarks, utilise un modèle **MLP pré-entraîné (Scikit-learn)** pour classifier la forme du visage, et renvoie la forme et les recommandations.
*   **Tests (Pytest)** : Système de tests complet qui vérifie la fonctionnalité de l'API, la précision du modèle et l'équité du système de recommandation.
*   **Méthodes alternatives** : Utilisation d'une méthode heuristique comme solution de secours et d'un clustering K-means comme approche alternative.

## 2. Diagramme d'Architecture Simplifié

```mermaid
graph TD
    A[Utilisateur] -- Interagit --> B(Frontend React);
    B -- Flux Vidéo --> C{Webcam};
    B -- Extrait Landmarks (TF.js Facemesh) --> D[Landmarks (468)];
    B -- Envoie Requête /predict --> E(Backend FastAPI);
    E -- Reçoit Landmarks --> F{Analyse Faciale MLP};
    F -- Charge Modèle .joblib / Scaler / Encoder --> G[Fichiers Modèles MLP];
    F -- Prédit Forme --> H[Forme du Visage (String)];
    E -- Récupère Recommandations --> I[Table Recommandations];
    E -- Renvoie Réponse --> B;
    B -- Affiche Forme & Recommandations --> A;

    J(Scripts de Test) -- Évaluent --> E;
    J -- Utilisent Données Test --> K[Données Simulées];
    J -- Mesurent Performance/Équité --> L[Rapports de Test];
```

## 3. Composants Principaux

### 3.1. Frontend (React + TensorFlow.js)

*   **Responsabilité :** Interface utilisateur, capture vidéo, extraction des landmarks.
*   **Technologie :** React, JavaScript, TensorFlow.js (@tensorflow-models/facemesh), WebRTC.
*   **Interaction :** Appelle le endpoint `/predict` du backend avec les landmarks.

### 3.2. Backend (FastAPI + Scikit-learn)

*   **Responsabilité :** API RESTful, chargement des modèles, prédiction de la forme du visage, gestion des recommandations.
*   **Technologie :** Python, FastAPI, Uvicorn, Scikit-learn, Joblib, Numpy, Pydantic.
*   **Modèle Principal :** MLP (Multi-Layer Perceptron) entraîné sur les coordonnées 2D des landmarks aplaties.
*   **Méthodes Alternatives :**
    * **Clustering K-means** pour une approche non supervisée.
    * **Méthode heuristique** basée sur des calculs de distances et ratios.
*   **Pré-traitement Prédiction :** Aplatissement des landmarks -> Scaling (StandardScaler) -> Prédiction -> Décodage (LabelEncoder).

### 3.3. Modèle d'Analyse Faciale (MLP)

*   **Objectif :** Classifier la forme du visage à partir des landmarks.
*   **Entraînement :** Réalisé sur des données simulées de landmarks.
*   **Artefacts Sauvegardés :**
    *   `mlp_model.joblib` : Le modèle MLP entraîné.
    *   `mlp_scaler.joblib` : Le `StandardScaler` ajusté.
    *   `mlp_label_encoder.joblib` : L'`LabelEncoder` ajusté.

### 3.4. Système de Tests

*   **Responsabilité :** Évaluation automatisée des performances et de la qualité du système.
*   **Tests Implémentés :**
    *   Tests fonctionnels de l'API.
    *   Tests unitaires des fonctions d'analyse faciale.
    *   Tests de biais et d'équité.
    *   Tests des utilitaires et des composants.

## 4. Flux de Données Principaux

### 4.1. Flux de Prédiction

1.  Frontend capture les landmarks (468 points) via TF.js.
2.  Frontend envoie les landmarks au Backend (`POST /predict`).
3.  Backend (`main.py`) reçoit la requête, valide les données (Pydantic).
4.  Backend appelle `estimate_face_shape_mlp` (`facial_analysis.py`).
5.  `estimate_face_shape_mlp` :
    a.  Vérifie si les modèles MLP sont chargés.
    b.  Convertit les landmarks en NumPy array, les aplatit.
    c.  Applique le `mlp_scaler` chargé.
    d.  Prédit avec `mlp_model.predict()`.
    e.  Décode le résultat avec `mlp_label_encoder`.
    f.  Retourne la forme (String).
6.  Backend (`main.py`) récupère les recommandations associées à la forme.
7.  Backend renvoie `PredictResponse` (forme, recommandations) au Frontend.
8.  Frontend affiche le résultat.

## 5. Pistes d'Amélioration et Sujets Avancés (Objectifs Futurs)

Cette section documente les axes d'amélioration potentiels.

### 5.1. Modèles Avancés

*   **CNN/Transformers :** Explorer des architectures plus complexes pour potentiellement améliorer la précision.
*   **Export ONNX :** Pour la portabilité et la performance cross-plateforme.
*   **Optimisation Modèles :** Lorsque la performance sera critique, investiguer la quantification, le pruning, ou la distillation de connaissances.

### 5.2. Pipeline Robuste Temps Réel

*   **Adaptation au Domaine :** Aller au-delà de simples données simulées. Explorer des techniques comme l'Augmentation de Données avancée.
*   **Optimisation Mobile/Embarqué :** Convertir et tester les modèles optimisés sur des plateformes cibles.
*   **Traitement Distribué/Parallélisé :** Si le backend devient un goulot d'étranglement, explorer l'utilisation de workers multiples.

### 5.3. Recommandation & Personnalisation

*   **Clustering Avancé :** Évaluer si des algorithmes de clustering plus sophistiqués appliqués aux landmarks pourraient révéler des sous-catégories de formes plus fines.
*   **Système de Recommandation Plus Intelligent :** Intégrer des facteurs comme les préférences utilisateur, l'historique d'achat ou des similarités entre produits.

### 5.4. Visualisation 3D et Réalité Augmentée

*   **Intégration Three.js :** Pour créer la scène 3D dans React.
*   **Positionnement & Tracking :** Aligner le modèle 3D des lunettes sur les landmarks faciaux.
*   **Rendu Réaliste :** Explorer les matériaux PBR (Physically Based Rendering) et l'éclairage dynamique.

## 6. Déploiement et Considérations Techniques

*   **Dépendances :** Gérées via `pyproject.toml` (backend) et `package.json` (frontend).
*   **Environnement :** Utilisation de `venv` pour isoler les dépendances Python.
*   **Serveur API :** Uvicorn comme serveur ASGI pour FastAPI.
*   **Déploiement Futur :** Pourrait être conteneurisé (Docker) et déployé sur des plateformes PaaS ou IaaS.
*   **CI/CD :** Non implémenté actuellement. Pourrait être ajouté pour automatiser les tests et les déploiements. 