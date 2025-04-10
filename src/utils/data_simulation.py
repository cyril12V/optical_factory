#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, get_args, Literal, Tuple

# --- Configuration du Chemin d'Import --- 
# Ajouter la racine du projet au PYTHONPATH pour trouver le module 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # utils -> src -> racine
sys.path.insert(0, str(PROJECT_ROOT))

# Gérer l'import de FaceShape plus gracieusement
# try:
#     from src.backend.app.facial_analysis import FaceShape 
#     FACE_SHAPE_AVAILABLE = True
# except (ImportError, ModuleNotFoundError) as e: # Capturer les deux erreurs possibles
#     print(f"(Avertissement dans data_simulation.py) Impossible d'importer FaceShape: {e}")
#     # Définir FaceShape comme Any pour permettre au reste du module de charger
#     # Les fonctions qui en dépendent directement pourraient échouer plus tard
#     FaceShape = Any 
#     FACE_SHAPE_AVAILABLE = False
#     # Ne pas appeler sys.exit(1) ici

# --- Fonctions Utilitaires pour la génération de données --- 

# Ajouter la constante FACE_SHAPES
FACE_SHAPES = ["Ovale", "Carrée", "Ronde", "Coeur", "Longue"]

def _generate_simulated_landmarks(shape_type: str, scale: float = 100.0) -> list:
    """Génère 468 landmarks simulés pour une forme donnée (simplifié)."""
    landmarks = [[0.0, 0.0, 0.0] for _ in range(468)]
    length = scale * 1.6
    cheek = scale * 1.0
    forehead = scale * 1.0
    jaw = scale * 1.0
    if shape_type == "Longue": length = scale * 1.8
    elif shape_type == "Carrée": length = scale * 1.0; forehead = scale * 1.0; jaw = scale * 1.0
    elif shape_type == "Ronde": length = scale * 1.0; jaw = scale * 0.8; cheek = scale * 1.1
    elif shape_type == "Coeur": forehead = scale * 1.1; jaw = scale * 0.8
    elif shape_type == "Ovale": length = scale * 1.3; jaw = scale * 0.9
    elif shape_type == "Inconnue": length, cheek, forehead, jaw = scale * 0.1, scale * 0.1, scale * 0.1, scale * 0.1
    # Ajouter une gestion pour les formes inconnues qui pourraient venir du CSV
    elif shape_type not in ["Ovale", "Carrée", "Ronde", "Coeur", "Longue", "Inconnue"]:
        print(f"(data_simulation) Avertissement: Forme '{shape_type}' inconnue pour la génération. Utilisation de valeurs par défaut.")
        length, cheek, forehead, jaw = scale, scale, scale, scale # Ou autre comportement par défaut

    try:
        landmarks[10] = [0.0, length / 2, 0.0]
        landmarks[152] = [0.0, -length / 2, 0.0]
        landmarks[234] = [-cheek / 2, 0.0, 0.0]
        landmarks[454] = [cheek / 2, 0.0, 0.0]
        landmarks[172] = [-jaw / 2, -length * 0.4, 0.0]
        landmarks[397] = [jaw / 2, -length * 0.4, 0.0]
        landmarks[103] = [-forehead / 2, length * 0.4, 0.0]
        landmarks[332] = [forehead / 2, length * 0.4, 0.0]
    except IndexError:
        print("Erreur: Index hors limites lors de la génération des landmarks simulés.")
        return [[0.0, 0.0, 0.0] for _ in range(468)]
    return [tuple(lm) for lm in landmarks]

# Ajouter la fonction generate_simulated_landmarks
def generate_simulated_landmarks(num_samples: int = 1000, include_all_shapes: bool = True) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Génère un jeu de données simulées de landmarks, leurs labels et un groupe.

    Args:
        num_samples (int): Nombre total d'échantillons à générer.
        include_all_shapes (bool): Si True, s'assure qu'au moins un échantillon
                                   de chaque forme dans FACE_SHAPES est inclus.

    Returns:
        tuple: Une liste de tableaux numpy de landmarks (chacun shape [468, 3]),
               une liste de labels string correspondants,
               une liste de groupes string ('A' ou 'B') correspondants.
    """
    landmarks_list = []
    labels_list = []
    groups_list = [] # Ajout de la liste des groupes
    available_shapes = list(FACE_SHAPES)
    possible_groups = ['A', 'B'] # Groupes possibles

    if include_all_shapes:
        # S'assurer qu'on a au moins un de chaque
        for shape in available_shapes:
            landmarks = _generate_simulated_landmarks(shape)
            # Ajouter un peu de bruit pour la variété
            noisy_landmarks = np.array(landmarks) + np.random.normal(0, 0.5, np.array(landmarks).shape)
            landmarks_list.append(noisy_landmarks)
            labels_list.append(shape)
            groups_list.append(random.choice(possible_groups)) # Assigner un groupe
            num_samples -= 1 # Décrémenter le nombre d'échantillons restants

    # Générer les échantillons restants aléatoirement
    for _ in range(num_samples):
        shape = random.choice(available_shapes)
        landmarks = _generate_simulated_landmarks(shape)
        # Ajouter un peu de bruit
        noisy_landmarks = np.array(landmarks) + np.random.normal(0, 0.5, np.array(landmarks).shape)
        landmarks_list.append(noisy_landmarks)
        labels_list.append(shape)
        groups_list.append(random.choice(possible_groups)) # Assigner un groupe

    # Mélanger les listes pour éviter les biais d'ordre
    combined = list(zip(landmarks_list, labels_list, groups_list))
    random.shuffle(combined)
    landmarks_list[:], labels_list[:], groups_list[:] = zip(*combined)

    return landmarks_list, labels_list, groups_list

# --- Chargement des Données depuis le CSV (en générant les landmarks) --- 
def load_test_data_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """Charge les données de test depuis un CSV, mais GÉNÈRE les landmarks."""
    data = []
    required_cols = ["id", "group", "expected_shape"]
    try:
        # Utiliser os.path.abspath pour gérer les chemins relatifs/absolus de manière robuste
        abs_filepath = os.path.abspath(filepath)
        if not os.path.exists(abs_filepath):
            raise FileNotFoundError(f"Le fichier CSV n'existe pas : {abs_filepath}")
            
        with open(abs_filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not all(col in reader.fieldnames for col in required_cols):
                 missing = set(required_cols) - set(reader.fieldnames or [])
                 print(f"ERREUR: Colonnes manquantes dans {abs_filepath}: {missing}")
                 return []
                 
            # >>> Supprimer la vérification basée sur l'import FaceShape
            # possible_shapes = get_args(FaceShape) if FACE_SHAPE_AVAILABLE else [] 
            
            for row in reader:
                try:
                    shape_to_generate = row['expected_shape']
                    # >>> Supprimer la vérification
                    # if FACE_SHAPE_AVAILABLE and shape_to_generate not in possible_shapes:
                    #     print(f"Avertissement: Forme attendue invalide '{shape_to_generate}' pour ID {row.get('id', 'N/A')}. Ligne ignorée.")
                    #     continue
                        
                    generated_landmarks = _generate_simulated_landmarks(shape_to_generate)
                    if len(generated_landmarks) != 468:
                         print(f"Erreur interne: La génération de landmarks n'a pas produit 468 points pour ID {row.get('id', 'N/A')}. Ligne ignorée.")
                         continue
                    data.append({
                        "id": row['id'],
                        "group": row['group'],
                        "expected_shape": shape_to_generate,
                        "landmarks": generated_landmarks
                    })
                except Exception as e_inner:
                     print(f"Erreur lors du traitement de la ligne ID {row.get('id', 'N/A')}: {e_inner}")
                     continue
                     
    except FileNotFoundError as e:
        print(f"ERREUR: {e}")
        return []
    except Exception as e_outer:
        print(f"Erreur inattendue lors de la lecture du CSV: {e_outer}")
        return []
    if not data:
         print(f"AVERTISSEMENT: Aucune donnée valide n'a pu être chargée/générée depuis {filepath}.")
    return data 