import pytest
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH pour trouver src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.backend.app.facial_analysis import get_recommendations, FaceShape, RECOMMENDATION_MAP
    from typing import get_args
except ImportError as e:
    print(f"Erreur d'import dans test_utils: {e}")
    # Optionnel: lever une exception si l'import est critique pour TOUS les tests du fichier
    # raise e 

# Récupérer les formes valides définies dans le Literal FaceShape (exclut "Inconnue")
VALID_SHAPES = [shape for shape in get_args(FaceShape) if shape != "Inconnue"]

def test_get_recommendations_valid_shapes():
    """Teste get_recommendations pour toutes les formes valides connues."""
    for shape in VALID_SHAPES:
        expected_recommendations = RECOMMENDATION_MAP.get(shape, [])
        actual_recommendations = get_recommendations(shape)
        # Vérifier que la liste retournée est correcte (l'ordre n'importe pas forcément)
        assert isinstance(actual_recommendations, list)
        assert set(actual_recommendations) == set(expected_recommendations), f"Erreur pour la forme {shape}"
        print(f"Test get_recommendations OK pour: {shape}")

def test_get_recommendations_unknown_shape():
    """Teste get_recommendations pour la forme "Inconnue"."""
    recommendations = get_recommendations("Inconnue")
    assert recommendations == [], "Devrait retourner une liste vide pour Inconnue"
    print("Test get_recommendations OK pour: Inconnue")

def test_get_recommendations_invalid_input():
    """Teste get_recommendations avec une chaîne invalide (non définie dans FaceShape)."""
    # La fonction est typée, mais testons le comportement au cas où
    # Elle devrait retourner la valeur par défaut de .get(), qui est [] ici.
    recommendations = get_recommendations("FormeInexistante") 
    assert recommendations == [], "Devrait retourner une liste vide pour une forme invalide"
    print("Test get_recommendations OK pour: Input invalide")

# Vous pouvez ajouter d'autres tests unitaires pour d'autres fonctions ici 