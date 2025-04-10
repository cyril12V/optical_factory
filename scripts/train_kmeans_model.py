#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import os
from pathlib import Path
import random
from collections import Counter
import joblib # Pour sauvegarder les modèles
import json # Pour sauvegarder le mapping

# --- Configuration du Chemin d'Import --- 
# Ajouter la racine du projet au PYTHONPATH pour trouver les modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Importer les fonctions/types nécessaires
    from src.backend.app.facial_analysis import FaceShape # Literal
    # !!! Attention: Importer depuis un dossier de tests n'est pas idéal pour la production
    # Il serait préférable de déplacer _generate_simulated_landmarks vers un module utilitaire
    from tests.backend.test_bias_fairness import _generate_simulated_landmarks
    # Importer depuis sklearn
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler # Important pour PCA/KMeans
except ModuleNotFoundError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que scikit-learn est installé (`pip install -r src/backend/requirements.txt`) et que le script est exécutable.")
    sys.exit(1)
except ImportError as e:
     print(f"Erreur d'importation: {e}")
     print("Problème potentiel avec l'import de _generate_simulated_landmarks depuis les tests.")
     sys.exit(1)

# --- Configuration de l'Entraînement --- 
# Formes valides à générer (exclure "Inconnue" pour l'entraînement du classifieur)
VALID_SHAPES = ["Ovale", "Carrée", "Ronde", "Coeur", "Longue"]
N_CLUSTERS = len(VALID_SHAPES) # Un cluster par forme valide
EXAMPLES_PER_SHAPE = 200 # Nombre d'exemples simulés par forme
PCA_N_COMPONENTS = 0.95 # Garder 95% de la variance (ou un nombre fixe ex: 50)
OUTPUT_MODEL_DIR = PROJECT_ROOT / "src" / "backend" / "models"
SCALER_FILE = OUTPUT_MODEL_DIR / "scaler.joblib"
PCA_FILE = OUTPUT_MODEL_DIR / "pca_model.joblib"
KMEANS_FILE = OUTPUT_MODEL_DIR / "kmeans_model.joblib"
CLUSTER_MAP_FILE = OUTPUT_MODEL_DIR / "cluster_to_shape_map.json"

# --- Fonctions Utilitaires --- 
def flatten_landmarks(landmarks_list: list) -> np.ndarray:
    """Aplatit une liste de listes de landmarks 3D en un seul vecteur 1D."""
    # Assurer que l'entrée est bien une liste de 468 points
    if not isinstance(landmarks_list, list) or len(landmarks_list) != 468:
        raise ValueError("Entrée landmarks invalide, attendu une liste de 468 points.")
    return np.array(landmarks_list).flatten()

# --- Script Principal d'Entraînement --- 
if __name__ == "__main__":
    print("--- Début de l'entraînement du modèle K-Means --- ")
    
    # 1. Génération des Données Simulées
    print(f"Génération de {EXAMPLES_PER_SHAPE} exemples pour {N_CLUSTERS} formes... ({EXAMPLES_PER_SHAPE * N_CLUSTERS} total)")
    all_landmarks_raw = []
    all_labels = []
    for shape in VALID_SHAPES:
        for i in range(EXAMPLES_PER_SHAPE):
            # Utiliser une échelle légèrement variable pour plus de diversité
            scale = random.uniform(80, 120)
            landmarks = _generate_simulated_landmarks(shape, scale=scale)
            if len(landmarks) == 468:
                all_landmarks_raw.append(landmarks)
                all_labels.append(shape)
            else:
                print(f"Avertissement: Échec de génération pour {shape}, exemple {i+1}")

    if not all_landmarks_raw:
        print("Erreur: Aucune donnée simulée n'a pu être générée.")
        sys.exit(1)

    print(f"Données générées: {len(all_landmarks_raw)} exemples.")

    # 2. Aplatir les Landmarks
    print("Aplatissement des landmarks...")
    try:
        X_flat = np.array([flatten_landmarks(lm) for lm in all_landmarks_raw])
    except ValueError as e:
        print(f"Erreur lors de l'aplatissement: {e}")
        sys.exit(1)
    print(f"Dimensions des données aplaties: {X_flat.shape}") # (N_exemples, 1404)

    # 3. Mise à l'échelle (Scaling)
    print("Mise à l'échelle des données (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # 4. Réduction de dimension (PCA)
    print(f"Application de PCA (gardant {PCA_N_COMPONENTS * 100}% de la variance)...")
    pca = PCA(n_components=PCA_N_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Dimensions après PCA: {X_pca.shape}")
    print(f"Variance expliquée par {pca.n_components_} composantes: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 5. Entraînement K-Means
    print(f"Entraînement de K-Means avec {N_CLUSTERS} clusters...")
    # Utiliser n_init='auto' pour une meilleure initialisation, random_state pour la reproductibilité
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    print("Entraînement K-Means terminé.")

    # 6. Déterminer le Mapping Cluster -> Forme
    print("Détermination du mapping Cluster -> Forme...")
    cluster_to_shape_map = {}
    for cluster_id in range(N_CLUSTERS):
        # Trouver les indices des points appartenant à ce cluster
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0:
            print(f"Avertissement: Le cluster {cluster_id} est vide!")
            cluster_to_shape_map[cluster_id] = "Inconnue" # Ou gérer autrement
            continue
        # Trouver la forme la plus fréquente parmi ces points
        labels_in_cluster = [all_labels[i] for i in indices]
        most_common_shape = Counter(labels_in_cluster).most_common(1)[0][0]
        cluster_to_shape_map[cluster_id] = most_common_shape
        print(f"  Cluster {cluster_id} -> {most_common_shape} (taille: {len(indices)})")
    
    # Vérifier si toutes les formes valides ont été mappées
    mapped_shapes = set(cluster_to_shape_map.values()) - {"Inconnue"}
    missing_shapes = set(VALID_SHAPES) - mapped_shapes
    if missing_shapes:
        print(f"Avertissement: Les formes suivantes n'ont pas été clairement associées à un cluster unique: {missing_shapes}")

    # 7. Sauvegarde des Modèles et du Mapping
    print(f"Sauvegarde des modèles dans {OUTPUT_MODEL_DIR}...")
    try:
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True) # Assurer que le dossier existe
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(pca, PCA_FILE)
        joblib.dump(kmeans, KMEANS_FILE)
        with open(CLUSTER_MAP_FILE, 'w') as f:
            json.dump(cluster_to_shape_map, f, indent=2)
        print("Modèles et mapping sauvegardés avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des modèles: {e}")
        sys.exit(1)

    print("--- Entraînement terminé --- ") 