#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import os
from pathlib import Path
import random
import joblib # Pour sauvegarder les modèles
import json # Pour sauvegarder le mapping (pour K-Means, pas MLP)
import logging
import time

# Imports pour l'export ONNX
# import skl2onnx
# import onnx
# from skl2onnx.common.data_types import FloatTensorType

# --- Configuration du Chemin d'Import --- 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.backend.app.facial_analysis import FaceShape # Literal
    from src.utils.data_simulation import _generate_simulated_landmarks, generate_simulated_data, FEATURES # Importer depuis utils
    # Importer depuis sklearn
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder # Ajouté LabelEncoder
    from sklearn.model_selection import train_test_split # Pour évaluer la performance
    from sklearn.metrics import accuracy_score # Pour évaluer la performance
    # from sklearn.ensemble import RandomForestClassifier # Supprimé
except ModuleNotFoundError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que scikit-learn est installé (`pip install -r src/backend/requirements.txt`) et que le script est exécutable.")
    sys.exit(1)

# --- Configuration de l'Entraînement --- 
VALID_SHAPES = ["Ovale", "Carrée", "Ronde", "Coeur", "Longue"]
EXAMPLES_PER_SHAPE = 300 # Augmenter un peu pour MLP?
TEST_SET_SIZE = 0.2 # Garder 20% des données pour tester la performance du modèle entraîné

OUTPUT_MODEL_DIR = PROJECT_ROOT / "src" / "backend" / "models"
MLP_SCALER_FILE = OUTPUT_MODEL_DIR / "mlp_scaler.joblib" # Nom différent pour éviter conflit avec K-Means
MLP_MODEL_FILE = OUTPUT_MODEL_DIR / "mlp_model.joblib"
MLP_ENCODER_FILE = OUTPUT_MODEL_DIR / "mlp_label_encoder.joblib" # Fichier pour l'encodeur

# Chemins pour RandomForest (utilise même scaler/encoder que MLP)
# RF_MODEL_FILE = OUTPUT_MODEL_DIR / "rf_model.joblib"

# --- Paramètres ---
NUM_SAMPLES = 2000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Paramètres MLP (inchangés)
MLP_HIDDEN_LAYER_SIZES = (128, 64)
MLP_MAX_ITER = 500
MLP_EARLY_STOPPING = True
MLP_N_ITER_NO_CHANGE = 20

# Paramètres RandomForest (exemple)
# RF_N_ESTIMATORS = 150 # Plus d'arbres
# RF_MAX_DEPTH = 20 # Limiter la profondeur
# RF_MIN_SAMPLES_SPLIT = 5
# RF_MIN_SAMPLES_LEAF = 2

# --- Fonctions Utilitaires --- 
def flatten_landmarks(landmarks_list: list) -> np.ndarray:
    """Aplatit une liste de listes de landmarks 3D en un seul vecteur 1D."""
    if not isinstance(landmarks_list, list) or len(landmarks_list) != 468:
        raise ValueError("Entrée landmarks invalide, attendu une liste de 468 points.")
    return np.array(landmarks_list).flatten()

# --- Script Principal d'Entraînement --- 
def main():
    logging.info("Début de l'entraînement du modèle MLP (uniquement)...")
    start_time = time.time()

    # 1. Générer les données simulées (inchangé)
    logging.info(f"Génération de {NUM_SAMPLES} échantillons de données simulées...")
    landmarks_data, labels = generate_simulated_data(NUM_SAMPLES)
    X = np.array([item for sublist in landmarks_data for item in sublist])
    y = np.array(labels)
    logging.info(f"Données générées. Forme X: {X.shape}, Forme y: {y.shape}")

    # 2. Encodage des labels (inchangé)
    logging.info("Encodage des labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logging.info(f"Labels encodés. Classes: {label_encoder.classes_}")

    # 3. Séparation Entraînement / Test (inchangé)
    logging.info("Séparation des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    logging.info(f"Taille Entraînement: {X_train.shape[0]}, Taille Test: {X_test.shape[0]}")

    # 4. Mise à l'échelle (StandardScaler) (inchangé)
    logging.info("Mise à l'échelle des données (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. Entraînement et Évaluation MLP --- (inchangé)
    logging.info("--- Entraînement MLP ---")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=MLP_EARLY_STOPPING,
        n_iter_no_change=MLP_N_ITER_NO_CHANGE,
        verbose=True
    )
    mlp_model.fit(X_train_scaled, y_train)
    logging.info("Entraînement MLP terminé.")

    # Évaluation MLP
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    logging.info(f"Performance MLP sur Test Set - Accuracy: {accuracy_mlp:.4f}")

    # --- Entraînement RandomForest supprimé --- 

    # --- 7. Sauvegarde des Artefacts (MLP uniquement) --- 
    logging.info("--- Sauvegarde des Artefacts MLP --- ")
    # Scaler et Encoder
    logging.info(f"Sauvegarde du scaler dans {MLP_SCALER_FILE}")
    joblib.dump(scaler, MLP_SCALER_FILE)
    logging.info(f"Sauvegarde de l'encodeur de labels dans {MLP_ENCODER_FILE}")
    joblib.dump(label_encoder, MLP_ENCODER_FILE)

    # Modèle MLP
    logging.info(f"Sauvegarde du modèle MLP dans {MLP_MODEL_FILE}")
    joblib.dump(mlp_model, MLP_MODEL_FILE)

    # Sauvegarde RandomForest supprimée

    # --- Export ONNX supprimé --- 

    end_time = time.time()
    logging.info(f"Script d'entraînement MLP terminé en {end_time - start_time:.2f} secondes.")

if __name__ == "__main__":
    main() 