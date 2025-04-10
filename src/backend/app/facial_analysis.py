import math
from typing import List, Tuple, Dict, Literal, get_args, TypeAlias
import joblib
import json
from pathlib import Path
import numpy as np
import logging

# Définir les types manquants
Landmarks: TypeAlias = List[List[float]] # Liste de [x, y, z]
FaceShape: TypeAlias = Literal["Ovale", "Carrée", "Ronde", "Coeur", "Longue", "Inconnue"] # Ajouter "Inconnue"

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition des chemins
BASE_DIR = Path(__file__).resolve().parent.parent # Remonte au dossier backend
MODEL_DIR = BASE_DIR / "models"
RECOMMENDATIONS_PATH = MODEL_DIR / "recommendations.json"

# --- Chemins Modèles MLP --- (Seuls les chemins MLP sont nécessaires maintenant)
MLP_SCALER_PATH = MODEL_DIR / "mlp_scaler.joblib"
MLP_MODEL_PATH = MODEL_DIR / "mlp_model.joblib"
MLP_ENCODER_PATH = MODEL_DIR / "mlp_label_encoder.joblib"

# Chemins K-Means (pour la compatibilité avec les tests)
KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_model.joblib"
PCA_MODEL_PATH = MODEL_DIR / "pca_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
CLUSTER_MAP_PATH = MODEL_DIR / "cluster_to_shape_map.json"

# Variables globales pour les modèles chargés
mlp_scaler = None
mlp_model = None
mlp_label_encoder = None
recommendations_data: Dict[str, List[str]] = {}

# Variables K-Means (pour la compatibilité avec les tests)
kmeans_models_loaded = False
kmeans_model = None
pca_model_kmeans = None
scaler_model_kmeans = None
cluster_to_shape_map = None

# --- Fonctions de chargement --- 

def load_recommendations():
    """Charge les données de recommandations depuis un fichier JSON."""
    global recommendations_data
    try:
        with open(RECOMMENDATIONS_PATH, 'r', encoding='utf-8') as f:
            import json
            recommendations_data = json.load(f)
        logger.info("Données de recommandations chargées avec succès.")
    except FileNotFoundError:
        logger.error(f"Erreur: Fichier de recommandations non trouvé à {RECOMMENDATIONS_PATH}")
        recommendations_data = {}
    except Exception as e:
        logger.error(f"Erreur lors du chargement des recommandations: {e}")
        recommendations_data = {}

def load_mlp_models():
    """Charge le scaler, le modèle MLP et l'encodeur de labels."""
    global mlp_scaler, mlp_model, mlp_label_encoder
    try:
        if MLP_SCALER_PATH.exists():
            mlp_scaler = joblib.load(MLP_SCALER_PATH)
            logger.info("Scaler MLP chargé.")
        else:
            logger.error(f"Scaler MLP non trouvé à {MLP_SCALER_PATH}")

        if MLP_MODEL_PATH.exists():
            mlp_model = joblib.load(MLP_MODEL_PATH)
            logger.info("Modèle MLP chargé.")
        else:
            logger.error(f"Modèle MLP non trouvé à {MLP_MODEL_PATH}")

        if MLP_ENCODER_PATH.exists():
            mlp_label_encoder = joblib.load(MLP_ENCODER_PATH)
            logger.info("Encodeur de labels MLP chargé.")
        else:
            logger.error(f"Encodeur MLP non trouvé à {MLP_ENCODER_PATH}")

    except Exception as e:
        logger.exception(f"Erreur lors du chargement des modèles MLP: {e}")

# --- Fonctions d'estimation --- 

def estimate_face_shape_from_landmarks_heuristic(landmarks: List[List[float]] | List[Tuple[float, float, float]]) -> str:
    """Estime la forme du visage en utilisant une heuristique basée sur les landmarks."""
    # (Votre logique heuristique ici - potentiellement simplifiée ou gardée)
    # Exemple simple basé sur la largeur vs hauteur (à adapter)
    if not landmarks or len(landmarks) < 153: # S'assurer qu'on a au moins les indices max utilisés
        return "Inconnue"
    try:
        # Convertir les tuples en tableau numpy
        points = np.array(landmarks)
        # Indices MediaPipe Facemesh pertinents (approximatifs)
        cheek_left_idx = 234 # Point externe pommette gauche
        cheek_right_idx = 454 # Point externe pommette droite
        forehead_top_idx = 10 # Milieu du front
        chin_bottom_idx = 152 # Pointe du menton
        jaw_left_idx = 172 # Point de la mâchoire gauche
        jaw_right_idx = 397 # Point de la mâchoire droite

        # Calcul des distances
        face_width_cheeks = np.linalg.norm(points[cheek_right_idx] - points[cheek_left_idx])
        face_width_jaw = np.linalg.norm(points[jaw_right_idx] - points[jaw_left_idx])
        face_height = np.linalg.norm(points[forehead_top_idx] - points[chin_bottom_idx])

        if face_width_cheeks == 0 or face_height == 0 or face_width_jaw == 0:
            return "Inconnue"

        # Ratios (exemples)
        height_ratio = face_height / face_width_cheeks
        jaw_ratio = face_width_jaw / face_width_cheeks

        # Logique de décision (exemple à affiner)
        if abs(height_ratio - 1.5) < 0.15: # Très allongé
             return "Longue"
        elif abs(face_width_cheeks - face_height) < face_width_cheeks * 0.1: # Presque aussi large que haut
            if jaw_ratio > 0.9: # Mâchoire large
                 return "Carrée"
            else: # Mâchoire plus étroite que pommettes
                 return "Ronde"
        elif height_ratio > 1.2: # Plus haut que large
            if jaw_ratio < 0.85: # Menton pointu
                 return "Coeur"
            else:
                 return "Ovale"
        else: # Default
            return "Ovale"

    except IndexError:
         logger.warning("IndexError dans l'heuristique, landmarks manquants ou indices incorrects.")
         return "Inconnue"
    except Exception as e:
        logger.error(f"Erreur inattendue dans estimate_face_shape_heuristic: {e}")
        return "Inconnue"

def estimate_face_shape_mlp(landmarks_flat: np.ndarray) -> str:
    """Estime la forme du visage en utilisant le modèle MLP pré-entraîné.

    Args:
        landmarks_flat (np.ndarray): Tableau NumPy aplati des landmarks, shape (1, 1404).

    Returns:
        str: La forme du visage prédite ("Ovale", "Carrée", etc.) ou "Inconnue"/"Erreur".
    """
    if mlp_model is None or mlp_scaler is None or mlp_label_encoder is None:
        logger.error("Modèles MLP non chargés. Impossible de prédire.")
        load_mlp_models() # Tentative de rechargement
        if mlp_model is None or mlp_scaler is None or mlp_label_encoder is None:
             return "Erreur: Modèles MLP non chargés"

    # Vérifier la forme de l'entrée directement (doit être (1, 1404))
    if not isinstance(landmarks_flat, np.ndarray) or landmarks_flat.shape != (1, 1404):
        logger.warning(f"Input landmarks array has invalid shape. Expected (1, 1404), got {getattr(landmarks_flat, 'shape', type(landmarks_flat))}")
        return "Inconnue"

    try:
        # 1. Données déjà aplaties
        landmarks_array = landmarks_flat # Renommer pour clarté ou utiliser directement

        # 2. Appliquer le scaler (qui attend 1404 features)
        landmarks_scaled = mlp_scaler.transform(landmarks_array)

        # 3. Prédiction (le modèle MLP s'attend aussi à 1404 features si le scaler a été fit dessus)
        prediction_encoded = mlp_model.predict(landmarks_scaled)

        # 4. Décoder le résultat
        predicted_shape = mlp_label_encoder.inverse_transform(prediction_encoded)[0]
        logger.info(f"Prédiction MLP: {predicted_shape}")
        return predicted_shape

    except Exception as e:
        logger.exception(f"Erreur lors de la prédiction MLP: {e}")
        return "Inconnue"

def get_recommendations(shape: str) -> List[str]:
    """Récupère les recommandations de lunettes pour une forme de visage donnée."""
    if not recommendations_data:
        logger.warning("Données de recommandations non chargées ou vides.")
        load_recommendations() # Tentative de rechargement
        if not recommendations_data:
             return ["Modèle Standard"] # Fallback ultime
    return recommendations_data.get(shape, ["Modèle Polyvalent"]) # Recommandation par défaut si forme inconnue

# --- Chargement initial des modèles et données ---
load_recommendations()
load_mlp_models() # Charger uniquement les modèles MLP au démarrage

logger.info("Module facial_analysis initialisé (Mode MLP).")

def flatten_landmarks_for_prediction(landmarks_list: list) -> np.ndarray:
    """Aplatit et reshape pour la prédiction (1 sample)."""
    if not isinstance(landmarks_list, list) or len(landmarks_list) != 468:
        raise ValueError("Entrée landmarks invalide")
    # Flatten puis reshape en (1, 1404) car les modèles attendent un batch
    return np.array(landmarks_list).flatten().reshape(1, -1)

# --- Fonction de Prédiction K-Means (inchangée) --- 
def estimate_face_shape_kmeans(points: Landmarks) -> FaceShape:
    """Estime la forme du visage en utilisant les modèles K-Means pré-entraînés."""
    if not kmeans_models_loaded or kmeans_model is None or pca_model_kmeans is None or scaler_model_kmeans is None or cluster_to_shape_map is None:
        print("Avertissement: Modèles K-Means non chargés, retour à Inconnue")
        return "Inconnue"
    try:
        X_flat = flatten_landmarks_for_prediction(points)
        X_scaled = scaler_model_kmeans.transform(X_flat) # Utiliser scaler K-Means
        X_pca = pca_model_kmeans.transform(X_scaled) # Utiliser PCA K-Means
        cluster_id = kmeans_model.predict(X_pca)[0] 
        predicted_shape = cluster_to_shape_map.get(cluster_id, "Inconnue")
        return predicted_shape
    except Exception as e:
        print(f"Erreur inattendue lors de la prédiction K-Means: {e}")
        return "Inconnue"

# --- Logique d'Estimation de Forme (Heuristique Affinée - Renommée) --- 
# Renommer l'ancienne fonction pour éviter les conflits
def estimate_face_shape_from_landmarks_heuristic(points: Landmarks) -> FaceShape:
    """
    Estime la forme du visage - Logique Heuristique Affinée avec Ratios Normalisés.
    NOTE: Toujours une heuristique, les seuils peuvent nécessiter ajustement.
    """
    if not points or len(points) < 468:
        return "Inconnue"
    try:
        # --- Indices Clés --- 
        # Verticaux
        p_top = points[10]      # Haut front
        p_chin = points[152]    # Menton
        # Largeurs
        p_cheek_l = points[234] # Pommette G
        p_cheek_r = points[454] # Pommette D
        p_jaw_l = points[172]   # Mâchoire G
        p_jaw_r = points[397]   # Mâchoire D
        p_forehead_l = points[103]# Front G
        p_forehead_r = points[332]# Front D
        # Optionnel: Yeux pour largeur inter-pupilles approx.
        # p_eye_l_inner = points[133]
        # p_eye_r_inner = points[362]
        # p_eye_l_outer = points[33] 
        # p_eye_r_outer = points[263]
        # Optionnel: Nez
        # p_nose_tip = points[1]
        # p_nose_bridge_top = points[6]
        # p_nose_l = points[129]
        # p_nose_r = points[358]

        # --- Calcul Distances --- 
        face_length = np.linalg.norm(points[10] - points[152])
        cheekbone_width = np.linalg.norm(points[234] - points[454])
        jawline_width = np.linalg.norm(points[172] - points[397])
        forehead_width = np.linalg.norm(points[103] - points[332])
        # inter_pupil_width = np.linalg.norm(points[33] - points[263]) # Ou inner
        # nose_width = np.linalg.norm(points[129] - points[358])

        # --- Vérification et Normalisation --- 
        dimensions = [face_length, cheekbone_width, jawline_width, forehead_width]
        if any(d <= 0 for d in dimensions): return "Inconnue"
        
        # Normaliser par la largeur des pommettes (référence)
        norm_length = face_length / cheekbone_width
        norm_forehead = forehead_width / cheekbone_width
        norm_jawline = jawline_width / cheekbone_width
        # cheekbone_width normalisé est 1 par définition

        # --- Logique de Classification Affinée (basée sur ratios normalisés) --- 
        
        # 1. Visage Long ? (Ratio longueur/largeur pommettes élevé)
        if norm_length > 1.65:
            return "Longue"

        # 2. Visage Carré ? (Long==Large, Front~Pommettes~Mâchoire)
        # Longueur proche largeur pommettes ET front/mâchoire proches largeur pommettes
        is_length_similar_width = abs(norm_length - 1.0) < 0.15 # Longueur +/- 15% de largeur pommettes
        is_forehead_similar_cheek = abs(norm_forehead - 1.0) < 0.15 # Front +/- 15% de largeur pommettes
        is_jawline_similar_cheek = abs(norm_jawline - 1.0) < 0.15 # Mâchoire +/- 15% de largeur pommettes
        if is_length_similar_width and is_forehead_similar_cheek and is_jawline_similar_cheek:
            return "Carrée"

        # 3. Visage en Coeur ? (Front large, Mâchoire étroite)
        # Front > Pommettes ET Mâchoire < Pommettes
        if norm_forehead > 1.02 and norm_jawline < 0.92:
            return "Coeur"

        # 4. Visage Rond ? (Largeur Pommettes dominante, Mâchoire étroite, Longueur~Largeur)
        # Pommettes >= Front ET Pommettes >= Mâchoire ET Mâchoire étroite ET Longueur proche Largeur
        # Note: cheekbone_width est la référence (norm = 1), donc on compare norm_forehead et norm_jawline à 1
        if norm_forehead <= 1.0 and norm_jawline <= 1.0 and norm_jawline < 0.92 and abs(norm_length - 1.0) < 0.18:
             return "Ronde"
        
        # 5. Visage Ovale (Cas par défaut / intermédiaire)
        # Longueur modérément > Largeur, Pommettes souvent les plus larges, Mâchoire plus étroite
        return "Ovale"

    except IndexError: return "Inconnue"
    except ZeroDivisionError: return "Inconnue"
    except Exception as e:
        print(f"Erreur inattendue dans estimate_face_shape_heuristic: {e}")
        return "Inconnue"

# --- Logique de Recommandation (Inchangée) --- 
RECOMMENDATION_MAP: Dict[FaceShape, List[str]] = {
    "Ovale": ["purple1", "classic_black", "aviator_gold", "round_tortoise"],
    "Carrée": ["classic_black", "aviator_gold", "round_tortoise"],
    "Ronde": ["modern_red", "round_tortoise"],
    "Coeur": ["purple1", "modern_red", "aviator_gold"],
    "Longue": ["classic_black"],
    "Inconnue": [] 
}
def get_recommendations(shape: FaceShape) -> List[str]:
    """
    Retourne une liste d'ID de lunettes recommandées pour une forme donnée.
    NOTE: Les recommandations sont basées sur des associations générales.
    """
    return RECOMMENDATION_MAP.get(shape, [])

# Supprimer/Commenter le code k-means s'il reste
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# FEATURE_INDICES = {...}
# kmeans_model = None
# scaler = None
# cluster_labels = {}
# def _extract_features(...): ...
# def classify_face_shape_kmeans(...): ...
# RECOMMENDATION_MAP_KMEANS = {...}
# def get_recommendations_kmeans(...): ... 