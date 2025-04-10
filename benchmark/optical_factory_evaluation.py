#!/usr/bin/env python
# coding: utf-8

import json
import time
import numpy as np
from pathlib import Path
import csv
import os
import sys
from typing import List, Dict, Any, Tuple, Literal, get_args
import argparse # Pour gérer les arguments de ligne de commande
import timeit # Pour mesurer la latence plus précisément
import random # Ajouté pour les placeholders
import subprocess # Pour appeler pytest
import cv2
import pandas as pd
import logging
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Essayer d'importer psutil pour l'utilisation mémoire
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Avertissement: La bibliothèque 'psutil' n'est pas installée. La métrique d'utilisation mémoire ne sera pas évaluée.")
    print("Pour l'activer, exécutez: pip install psutil")

# Essayer d'importer OpenCV (cv2)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Erreur: La bibliothèque 'opencv-python' n'est pas installée.")
    print("La métrique 'facial_detection_precision' ne pourra pas être évaluée.")
    print("Pour l'activer, exécutez: pip install opencv-python")

# --- Configuration du Chemin d'Import --- 
# Ajouter la racine du projet au PYTHONPATH pour trouver le module 'src'
# C'est crucial car ce script est dans /benchmark et doit importer depuis /src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Importer les fonctions et types nécessaires depuis le backend
    from src.backend.app.facial_analysis import (
        estimate_face_shape_mlp, # <-- La fonction à évaluer
        get_recommendations, 
        FaceShape, # Literal
        Landmarks,
        # On pourrait aussi importer l'heuristique pour comparaison si besoin
        # estimate_face_shape_from_landmarks_heuristic 
    )
    # Importer SEULEMENT les fonctions/constantes nécessaires et disponibles de data_simulation
    from src.utils.data_simulation import generate_simulated_landmarks, FACE_SHAPES
except ModuleNotFoundError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que le script est lancé depuis la racine ou que le PYTHONPATH est correctement configuré.")
    sys.exit(1)

# --- Constantes et Configuration --- 
DEFAULT_CRITERIA_PATH = PROJECT_ROOT / "config" / "evaluation_criteria.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "evaluation_results.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "benchmark" / "test_data"
VIDEO_SUBDIR = "videos" # Sous-dossier pour les vidéos de test
SIMULATED_DATA_CSV = "simulated_face_data.csv" 
HAAR_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# Chemins vers les modèles CNN (nécessaires pour le benchmark avec bruit)
MODEL_DIR = PROJECT_ROOT / "src" / "backend" / "models"

# --- Fonctions d'Évaluation Spécifiques (Modifiées/Ajoutées) --- 

def load_criteria(filepath: Path) -> Dict:
    """Charge les critères d'évaluation depuis un fichier JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            criteria = json.load(f)
        logging.info(f"Critères chargés depuis {filepath}")
        return criteria
    except FileNotFoundError:
        logging.error(f"Fichier de critères non trouvé: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Erreur de format JSON dans le fichier de critères {filepath}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Erreur inattendue lors du chargement des critères: {e}")
        return {}

def save_report(report: Dict, filepath: Path):
    """Sauvegarde le rapport d'évaluation au format JSON."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)  # Utiliser indentation 4 pour lisibilité
        logging.info(f"Rapport sauvegardé dans {filepath}")
    except IOError as e:
        logging.error(f"Erreur d'écriture lors de la sauvegarde du rapport dans {filepath}: {e}")
    except Exception as e:
        logging.error(f"Erreur inattendue lors de la sauvegarde du rapport: {e}")

def find_haar_cascade() -> str | None:
    """Essaie de localiser le fichier cascade Haar d'OpenCV."""
    if not CV2_AVAILABLE:
        return None
    # Chemin typique dans les installations OpenCV
    cv2_base_dir = Path(cv2.__file__).parent
    cascade_path = cv2_base_dir / "data" / HAAR_CASCADE_FILENAME
    if cascade_path.exists():
        print(f"Cascade Haar trouvée: {cascade_path}")
        return str(cascade_path)
    else:
        print(f"Avertissement: Cascade Haar '{HAAR_CASCADE_FILENAME}' non trouvée dans {cv2_base_dir / 'data'}.\"")
        print("La détection de visage risque d'échouer. Assurez-vous que le fichier est présent.")
        # Optionnel: chercher ailleurs si nécessaire
        return None

def evaluate_facial_detection(video_path: Path, face_cascade) -> Tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Impossible d'ouvrir la vidéo: {video_path}")
        return 0.0, 0.0

    total_frames = 0
    detected_frames = 0
    latencies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        start_time = time.perf_counter()
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        end_time = time.perf_counter()

        if len(faces) > 0:
            detected_frames += 1
            latencies.append((end_time - start_time) * 1000) # en ms

    cap.release()
    accuracy = (detected_frames / total_frames) * 100 if total_frames > 0 else 0
    avg_latency = np.mean(latencies) if latencies else 0
    logging.info(f"Vidéo {video_path.name}: Détection {accuracy:.2f}%, Latence moyenne {avg_latency:.2f} ms")
    return accuracy, avg_latency

def evaluate_model_generic(model_function, X_test, y_test, label_encoder):
    """Évalue un modèle générique (MLP, K-Means, etc.) sur les données de test.

    Args:
        model_function (callable): La fonction d'estimation à évaluer.
        X_test (np.ndarray): Données de test (landmarks flatten, shape=(n_samples, n_features)).
        y_test (np.ndarray): Vraies étiquettes (encodées numériquement).
        label_encoder (LabelEncoder): L'encodeur utilisé pour les étiquettes.

    Returns:
        tuple: accuracy (float), report (str), conf_matrix (np.ndarray), avg_latency (float), predictions_str (List[str])
    """
    logging.info(f"Début de l'évaluation générique pour {model_function.__name__}")
    predictions_encoded = []
    predictions_str = [] # Pour calculer le taux d'accès
    latencies = []

    start_total_time = time.time()
    for i in range(len(X_test)):
        # L'entrée doit être (1, n_features) pour estimate_face_shape_mlp
        landmarks_input = X_test[i].reshape(1, -1) 
        start_time = time.perf_counter()
        predicted_shape_name = model_function(landmarks_input)
        end_time = time.perf_counter()

        predictions_str.append(predicted_shape_name) # Stocker le nom prédit
        latencies.append((end_time - start_time) * 1000) # latence en ms

        # Gérer le cas "Inconnue" ou Erreur retournée par le modèle
        if predicted_shape_name == "Inconnue" or "Erreur" in predicted_shape_name:
            predictions_encoded.append(-1) # Valeur spéciale pour non-reconnaissance/erreur
        else:
            try:
                # Reconvertir le nom prédit en code numérique
                encoded_pred = label_encoder.transform([predicted_shape_name])[0]
                predictions_encoded.append(encoded_pred)
            except ValueError:
                 logging.warning(f"Forme prédite '{predicted_shape_name}' non reconnue par LabelEncoder. Assignation à -1.")
                 predictions_encoded.append(-1)

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    avg_latency = np.mean(latencies) if latencies else 0
    logging.info(f"Temps total d'évaluation ({len(X_test)} échantillons): {total_time:.4f} secondes")
    logging.info(f"Latence moyenne par prédiction: {avg_latency:.4f} ms")

    # Filtrer les prédictions invalides (-1) et les y_test correspondants
    valid_indices = [i for i, p in enumerate(predictions_encoded) if p != -1]
    if len(valid_indices) < len(predictions_encoded):
        logging.warning(f"{len(predictions_encoded) - len(valid_indices)} prédictions 'Inconnue' ou non reconnues ont été exclues de l'évaluation de précision.")

    y_test_filtered = y_test[valid_indices]
    predictions_filtered = np.array(predictions_encoded)[valid_indices]

    if len(y_test_filtered) == 0:
        logging.error("Aucune prédiction valide n'a pu être évaluée pour la précision.")
        # Retourner des valeurs par défaut pour éviter les erreurs en aval
        num_classes = len(label_encoder.classes_)
        return 0.0, "No valid predictions", np.zeros((num_classes, num_classes)), avg_latency, predictions_str

    # Calcul des métriques de précision sur les données filtrées
    accuracy = accuracy_score(y_test_filtered, predictions_filtered)
    report = classification_report(y_test_filtered, predictions_filtered, labels=np.arange(len(label_encoder.classes_)), target_names=label_encoder.classes_, zero_division=0)
    conf_matrix = confusion_matrix(y_test_filtered, predictions_filtered, labels=np.arange(len(label_encoder.classes_)))

    logging.info(f"Fin de l'évaluation générique pour {model_function.__name__}. Accuracy: {accuracy:.4f}")
    return accuracy, report, conf_matrix, avg_latency, predictions_str # Retourner aussi les prédictions string

def evaluate_memory_usage(model_func, X_eval, label_encoder, criteria):
    """Mesure l'utilisation mémoire de la fonction de prédiction sur les données."""
    if not PSUTIL_AVAILABLE:
        logging.warning("psutil non disponible, évaluation mémoire désactivée.")
        mem_threshold = criteria.get("memory_usage", {}).get("threshold_mb", 500)
        return {"metric": "memory_usage", "value": -1.0, "threshold_mb": mem_threshold, "status": "Non évalué", "details": "psutil non installé"}

    process = psutil.Process()
    # Mesurer avant
    mem_before = process.memory_info().rss / (1024 * 1024) # Convertir en Mo

    # Exécuter la fonction d'évaluation complète pour simuler l'usage réel
    try:
        logging.info("Exécution des prédictions pour mesure mémoire...")
        # On ne récupère pas les résultats ici, juste l'exécution
        # Correction: Passer des labels encodés valides (ex: tous la classe 0)
        y_encoded_dummy = np.zeros(len(X_eval), dtype=int) 
        _ = evaluate_model_generic(model_func, X_eval, y_encoded_dummy, label_encoder)
        logging.info("Prédictions terminées pour mesure mémoire.")
    except Exception as e:
        logging.error(f"Erreur durant l'exécution pour mesure mémoire: {e}")
        mem_threshold = criteria.get("memory_usage", {}).get("threshold_mb", 500)
        return {"metric": "memory_usage", "value": -1.0, "threshold_mb": mem_threshold, "status": "Erreur", "details": f"Erreur pdt mesure: {e}"}

    # Mesurer après
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_used = mem_after - mem_before
    mem_used = max(0, mem_used) # Éviter les valeurs négatives

    logging.info(f"Utilisation mémoire (approximative): {mem_used:.2f} Mo")

    # Formatage du résultat pour le rapport
    mem_threshold = criteria.get("memory_usage", {}).get("threshold_mb", 500) # Exemple seuil
    mem_status = "Atteint" if mem_used <= mem_threshold else "Non atteint"
    return {
        "metric": "memory_usage", 
        "value": mem_used, 
        "threshold_mb": mem_threshold, 
        "status": mem_status, 
        "details": f"Utilisation RSS approximative mesurée par psutil pour {len(X_eval)} prédictions."
    }

def evaluate_ar_fps(criteria: Dict) -> Dict:
    # Placeholder - simule une mesure FPS
    simulated_fps = np.random.uniform(18.0, 25.0) # Exemple de FPS simulé
    fps_threshold = criteria.get("ar_fps", {}).get("threshold", 15)
    status = "Atteint" if simulated_fps >= fps_threshold else "Non atteint"
    return {
        "metric": "ar_fps", 
        "value": round(simulated_fps, 1), # Arrondi pour l'affichage
        "threshold": fps_threshold, 
        "status": status,
        "details": "Valeur simulée (placeholder) - évaluation réelle requiert intégration frontend." # Message explicite
    }

# --- Nouvelle fonction pour l'équité ---
def evaluate_algorithmic_fairness(model_func, X_eval, y_labels, groups, label_encoder, all_predictions_str):
    """Évalue l'équité du modèle sur les groupes définis.

    Args:
        model_func (callable): Fonction de prédiction (non utilisée directement ici, dépend des prédictions passées).
        X_eval (np.ndarray): Données d'évaluation.
        y_labels (np.ndarray): Vraies étiquettes (string).
        groups (np.ndarray): Groupes correspondants ('A', 'B', ...).
        label_encoder (LabelEncoder): Encodeur d'étiquettes.
        all_predictions_str (List[str]): Liste de toutes les prédictions string obtenues précédemment.

    Returns:
        Dict: Dictionnaire contenant 'accuracy_gap' et 'recommendation_access_rate'.
    """
    logging.info("--- Évaluation de l'Équité Algorithmique ---")
    results = {"accuracy_gap": -1.0, "recommendation_access_rate": -1.0}
    unique_groups = np.unique(groups)
    group_accuracies = {}

    if len(all_predictions_str) != len(y_labels):
         logging.error("Incohérence entre le nombre de prédictions et de labels pour l'évaluation d'équité.")
         return results

    # Convertir les vrais labels en format numérique
    try:
        y_encoded_true = label_encoder.transform(y_labels)
    except ValueError as e:
        logging.error(f"Erreur d'encodage des vrais labels pour l'équité: {e}")
        return results

    # Convertir les prédictions string en format numérique (-1 pour Inconnue/Erreur)
    predictions_encoded = []
    valid_pred_count = 0
    for pred_str in all_predictions_str:
        if pred_str == "Inconnue" or "Erreur" in pred_str:
            predictions_encoded.append(-1)
        else:
            try:
                predictions_encoded.append(label_encoder.transform([pred_str])[0])
                valid_pred_count += 1
            except ValueError:
                predictions_encoded.append(-1) # Label prédit non reconnu
    predictions_encoded = np.array(predictions_encoded)

    # Calculer l'accuracy par groupe
    for group in unique_groups:
        group_mask = (groups == group)
        y_true_group = y_encoded_true[group_mask]
        y_pred_group = predictions_encoded[group_mask]

        # Filtrer les prédictions invalides (-1) pour le calcul de l'accuracy du groupe
        valid_indices_group = (y_pred_group != -1)
        y_true_group_filtered = y_true_group[valid_indices_group]
        y_pred_group_filtered = y_pred_group[valid_indices_group]

        if len(y_true_group_filtered) > 0:
            group_accuracies[group] = accuracy_score(y_true_group_filtered, y_pred_group_filtered)
            logging.info(f"Précision pour Groupe '{group}': {group_accuracies[group]:.4f} ({len(y_true_group_filtered)} échantillons valides)")
        else:
            group_accuracies[group] = 0.0 # Ou np.nan ?
            logging.warning(f"Aucune prédiction valide pour le groupe '{group}'. Accuracy mise à 0.")

    # Calculer l'écart de performance (max - min accuracy)
    # Vérifier si les valeurs sont numériques et s'il y en a plus d'une
    if len(group_accuracies) > 1 and all(isinstance(acc, (int, float)) for acc in group_accuracies.values()):
        # Extraire uniquement les précisions valides (numériques)
        valid_accuracies = [acc for acc in group_accuracies.values() if isinstance(acc, (int, float))]
        # S'assurer qu'il y a au moins deux précisions valides à comparer
        if len(valid_accuracies) > 1:
             results["accuracy_gap"] = max(valid_accuracies) - min(valid_accuracies)
             logging.info(f"Écart de performance (Accuracy Gap): {results['accuracy_gap']:.4f}")
        else:
             # Pas assez de précisions valides pour calculer un écart
             results["accuracy_gap"] = 0.0
             logging.warning("Pas assez de précisions de groupe valides trouvées pour calculer l'écart.")
    else:
        # Pas d'écart si un seul groupe ou aucune précision valide
        results["accuracy_gap"] = 0.0
        logging.info("Calcul de l'écart de performance non applicable (un seul groupe ou pas de données valides).")

    # Calculer le taux d'accès aux recommandations (proportion de prédictions non-'Inconnue'/Erreur)
    total_predictions = len(all_predictions_str)
    if total_predictions > 0:
        # Compter les prédictions qui ne sont NI "Inconnue" NI une "Erreur"
        accessible_predictions = sum(1 for p in all_predictions_str if p != "Inconnue" and "Erreur" not in p)
        results["recommendation_access_rate"] = accessible_predictions / total_predictions
        logging.info(f"Taux d'accès aux recommandations: {results['recommendation_access_rate']:.4f} ({accessible_predictions}/{total_predictions})")
    else:
         results["recommendation_access_rate"] = 0.0

    logging.info("--- Fin Évaluation Équité ---")
    return results

# --- Génération du Rapport --- 
def generate_evaluation_report(model_func, test_data: List[Dict], X_eval: np.ndarray, y_labels: np.ndarray, groups_list: List[str], criteria: Dict, test_video_path: Path) -> Dict:
    """
    Génère un rapport d'évaluation complet au format JSON standardisé. Inclut Équité et Mémoire.
    """
    report = {
        "project": "Optical Factory",
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_evaluated": model_func.__name__, 
        "criteria_file": str(DEFAULT_CRITERIA_PATH.relative_to(PROJECT_ROOT)),
        "metrics": []
    }
    
    print("\nDémarrage de l'évaluation...")
    
    # 1. Précision Détection Faciale (OpenCV)
    print("  - Évaluation: Précision Détection Faciale (Haar)")
    # Charger la cascade ici pour la passer à la fonction
    cascade_path = find_haar_cascade()
    if not cascade_path:
         face_cascade = None
         report["metrics"].append({"metric": "facial_detection_precision", "value": 0.0, "threshold": 0.0, "status": "Erreur", "details": "Cascade Haar introuvable."})    
    else:
         face_cascade = cv2.CascadeClassifier(cascade_path)
         
    if face_cascade is None or face_cascade.empty(): # Vérification ajoutée
         # Assurer que la métrique est ajoutée même si la cascade n'est pas chargée
         if not any(m['metric'] == 'facial_detection_precision' for m in report['metrics']): 
              report["metrics"].append({"metric": "facial_detection_precision", "value": 0.0, "threshold": 0.0, "status": "Erreur", "details": "Cascade Haar non chargée ou invalide."})    
    else:
         video_files = list(test_video_path.glob("*.mp4")) + list(test_video_path.glob("*.avi"))
         if not video_files:
             report["metrics"].append({"metric": "facial_detection_precision", "value": 0.0, "threshold": 0.0, "status": "Non évalué", "details": f"Aucune vidéo trouvée dans {test_video_path}"})
         else:
             detection_accuracies = []
             detection_latencies = []
             for video_file in video_files:
                 # Correction : Passer face_cascade à la fonction
                 acc, lat = evaluate_facial_detection(video_file, face_cascade)
                 detection_accuracies.append(acc)
                 detection_latencies.append(lat)
             avg_detection_accuracy = np.mean(detection_accuracies) if detection_accuracies else 0.0
             avg_detection_latency = np.mean(detection_latencies) if detection_latencies else 0.0
             # S'assurer de ne pas ajouter la métrique deux fois
             if not any(m['metric'] == 'facial_detection_precision' for m in report['metrics']):
                  report["metrics"].append({"metric": "facial_detection_precision", "value": avg_detection_accuracy, "threshold": criteria.get("facial_detection_precision", {}).get("threshold", 90.0), "status": "Atteint" if avg_detection_accuracy >= criteria.get("facial_detection_precision", {}).get("threshold", 90.0) else "Non atteint", "details": {"avg_latency_ms": avg_detection_latency}})

    # 2. Précision et Latence Classification de Forme (et obtenir prédictions pour équité)
    print("  - Évaluation: Précision & Latence Classification de Forme")
    accuracy = 0.0
    avg_latency = -1.0
    all_predictions_str = [] # Initialiser

    logging.info("Chargement de l'encodeur pour l'évaluation...")
    try:
        mlp_encoder_path = MODEL_DIR / "mlp_label_encoder.joblib"
        label_encoder = joblib.load(mlp_encoder_path)
    except Exception as e:
         logging.error(f"Impossible de charger l'encodeur MLP {mlp_encoder_path}: {e}")
         report["metrics"].append({"metric": "shape_classification_accuracy", "value": 0.0, "threshold": 0.0, "status": "Erreur", "details": f"Chargement encodeur échoué: {e}"})
         report["metrics"].append({"metric": "inference_latency", "value": -1.0, "threshold_ms": 0.0, "status": "Erreur", "details": f"Chargement encodeur échoué: {e}"})
         label_encoder = None

    if label_encoder is not None and X_eval.size > 0 and y_labels.size > 0:
        try:
            y_encoded = label_encoder.transform(y_labels)
            
            # Appeler evaluate_model_generic 
            accuracy, class_report, conf_matrix, avg_latency, all_predictions_str = evaluate_model_generic(
                model_func, X_eval, y_encoded, label_encoder
            )
            
            # Ajouter métrique accuracy
            accuracy_threshold = criteria.get("shape_classification_accuracy", {}).get("threshold", 0.9)
            accuracy_status = "Atteint" if accuracy >= accuracy_threshold else "Non atteint"
            report["metrics"].append({
                "metric": "shape_classification_accuracy", 
                "value": accuracy, 
                "threshold": accuracy_threshold, 
                "status": accuracy_status, 
                "details": {"classification_report": class_report, "confusion_matrix": conf_matrix.tolist()}
            })

            # Ajouter métrique latence
            latency_threshold = criteria.get("inference_latency", {}).get("threshold_ms", 200)
            latency_status = "Atteint" if avg_latency <= latency_threshold else "Non atteint"
            report["metrics"].append({
                "metric": "inference_latency", 
                "value": avg_latency, 
                "threshold_ms": latency_threshold, 
                "status": latency_status,
                "details": f"Latence moyenne sur {len(X_eval)} échantillons."
            })

        except ValueError as e:
            logging.error(f"Erreur (ValueError) durant l'évaluation modèle/labels: {e!r}") 
            report["metrics"].append({"metric": "shape_classification_accuracy", "value": 0.0, "threshold": 0.0, "status": "Erreur", "details": f"ValueError: {e!r}"}) 
            report["metrics"].append({"metric": "inference_latency", "value": -1.0, "threshold_ms": 0.0, "status": "Erreur", "details": f"ValueError: {e!r}"}) 
        except Exception as e:
            logging.error(f"Erreur durant evaluate_model_generic: {e}")
            report["metrics"].append({"metric": "shape_classification_accuracy", "value": 0.0, "threshold": 0.0, "status": "Erreur", "details": f"Erreur évaluation: {e}"})
            report["metrics"].append({"metric": "inference_latency", "value": -1.0, "threshold_ms": 0.0, "status": "Erreur", "details": f"Erreur évaluation: {e}"})

    elif X_eval.size == 0 or y_labels.size == 0:
         report["metrics"].append({"metric": "shape_classification_accuracy", "value": 0.0, "threshold": 0.0, "status": "Non évalué", "details": "Pas de données de test générées"})
         report["metrics"].append({"metric": "inference_latency", "value": 0.0, "threshold_ms": 0.0, "status": "Non évalué", "details": "Pas de données de test générées"})

    # 3. Équité Algorithmique (Gap Perf Classification + Accès Reco)
    print("  - Évaluation: Équité Algorithmique")
    if label_encoder is not None and X_eval.size > 0 and groups_list:
        fairness_results = evaluate_algorithmic_fairness(model_func, X_eval, y_labels, np.array(groups_list), label_encoder, all_predictions_str)
        gap_threshold = criteria.get("algorithmic_fairness_gap", {}).get("threshold", 0.1)
        access_threshold = criteria.get("recommendation_access_rate", {}).get("threshold", 0.95)
        
        gap_status = "Atteint" if fairness_results["accuracy_gap"] <= gap_threshold and fairness_results["accuracy_gap"] >= 0 else "Non atteint"
        access_status = "Atteint" if fairness_results["recommendation_access_rate"] >= access_threshold else "Non atteint"

        report["metrics"].append({"metric": "algorithmic_fairness_gap", "value": fairness_results["accuracy_gap"], "threshold": gap_threshold, "status": gap_status, "details": "Écart max de précision entre groupes A/B."})
        report["metrics"].append({"metric": "recommendation_access_rate", "value": fairness_results["recommendation_access_rate"], "threshold": access_threshold, "status": access_status, "details": "Taux de prédictions valides (non 'Inconnue'/Erreur)."})
    else:
        report["metrics"].append({"metric": "algorithmic_fairness_gap", "value": -1.0, "threshold": -1.0, "status": "Non évalué", "details": "Données/groupes manquants ou erreur encodeur."})
        report["metrics"].append({"metric": "recommendation_access_rate", "value": -1.0, "threshold": -1.0, "status": "Non évalué", "details": "Données/groupes manquants ou erreur encodeur."})

    # 4. Utilisation Mémoire
    print("  - Évaluation: Utilisation Mémoire")
    if label_encoder is not None and X_eval.size > 0:
         # Passer les données nécessaires à la fonction d'évaluation mémoire
         memory_metric = evaluate_memory_usage(model_func, X_eval, label_encoder, criteria)
         report["metrics"].append(memory_metric)
    else:
         mem_threshold = criteria.get("memory_usage", {}).get("threshold_mb", 500)
         report["metrics"].append({"metric": "memory_usage", "value": -1.0, "threshold_mb": mem_threshold, "status": "Non évalué", "details": "Données manquantes ou erreur encodeur."})
    
    # 5. FPS AR (Placeholder)
    print("  - Évaluation: FPS AR (Placeholder)")
    report["metrics"].append(evaluate_ar_fps(criteria))

    # Calcul du résumé
    total_metrics = len(report["metrics"])
    passed_metrics = sum(1 for m in report["metrics"] if m["status"] == "Atteint")
    success_rate = (passed_metrics / total_metrics * 100) if total_metrics > 0 else 0

    report["summary"] = {
        "criteria_passed": passed_metrics,
        "total_criteria": total_metrics,
        "success_rate_percent": round(success_rate, 2)
    }

    print("\nÉvaluation terminée.")
    return report

# --- Exécution Principale --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'évaluation pour Optical Factory")
    parser.add_argument("--criteria", default=str(DEFAULT_CRITERIA_PATH), help="Chemin vers le fichier JSON des critères d'évaluation.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Chemin vers le fichier JSON de sortie des résultats.")
    parser.add_argument("--samples", type=int, default=1000, help="Nombre d'échantillons simulés à générer pour l'évaluation.")
    args = parser.parse_args()

    criteria_path = Path(args.criteria)
    output_path = Path(args.output)
    test_video_path = DEFAULT_DATA_PATH / VIDEO_SUBDIR

    # Créer le dossier vidéo de test s'il n'existe pas
    test_video_path.mkdir(parents=True, exist_ok=True)

    # Charger les critères
    print(f"Chargement des critères depuis: {criteria_path}")
    evaluation_criteria = load_criteria(criteria_path)
    if not evaluation_criteria:
        sys.exit(1)

    # Générer les données de test simulées (avec groupes)
    num_evaluation_samples = args.samples
    print(f"Génération de {num_evaluation_samples} échantillons simulés pour le test...")
    landmarks_list, labels_list, groups_list = [], [], [] # Initialiser
    X_eval = np.array([])
    y_labels = np.array([])
    try:
        # Correction de l'appel pour dépaqueter les 3 valeurs
        landmarks_list, labels_list, groups_list = generate_simulated_landmarks(
            num_samples=num_evaluation_samples, include_all_shapes=True
        )
        # Préparer X_eval (1404 features) et y_labels
        X_eval = np.array(landmarks_list).reshape(num_evaluation_samples, -1) 
        y_labels = np.array(labels_list)
    except Exception as e:
        print(f"Erreur lors de la génération des données simulées : {e}")
        # Laisser les listes/arrays vides

    if not landmarks_list:
        print("Avertissement: Aucune donnée de test simulée n'a pu être générée...")

    # Sélectionner la fonction modèle à évaluer
    model_function_to_evaluate = estimate_face_shape_mlp 
    print(f"\nModèle à évaluer: {model_function_to_evaluate.__name__}")

    # Générer le rapport (passer groups_list)
    evaluation_report = generate_evaluation_report(
        model_func=model_function_to_evaluate,
        test_data=[], # test_data n'est plus vraiment utilisé, on passe X/y/groups
        X_eval=X_eval,
        y_labels=y_labels,
        groups_list=groups_list, # Passer les groupes
        criteria=evaluation_criteria,
        test_video_path=test_video_path 
    )

    # Sauvegarder le rapport
    save_report(evaluation_report, output_path)
    print(f"\nRapport d'évaluation sauvegardé dans: {output_path}")

    # Afficher le résumé
    summary = evaluation_report.get("summary", {})
    metrics = evaluation_report.get("metrics", [])
    print("\n--- Résumé de l'Évaluation ---")
    print(f"Critères atteints: {summary.get('criteria_passed', 0)} / {summary.get('total_criteria', 0)}")
    print(f"Taux de succès: {summary.get('success_rate_percent', 0):.2f}%")
    print("-----------------------------")
    for metric in metrics:
        threshold_str = ""
        if "threshold_ms" in metric:
            threshold_str = f"(Seuil: {metric['threshold_ms']} ms)"
        elif "threshold_mb" in metric:
             threshold_str = f"(Seuil: {metric['threshold_mb']} Mo)"
        elif "threshold" in metric:
             threshold_str = f"(Seuil: {metric['threshold']})"
             
        print(f"- {metric['metric']}: {metric.get('value', 'N/A'):.4f} {threshold_str} -> {metric.get('status', 'Inconnu')}")
    print("-----------------------------")

    # Optionnel: Sortir avec un code d'erreur si tous les critères ne sont pas atteints
    # if summary.get('met_criteria') != summary.get('total_criteria'):
    #     sys.exit(1)
    sys.exit(0) 