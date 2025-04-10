#!/usr/bin/env python
# coding: utf-8

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from typing import List, Tuple
import random
from unittest.mock import patch # Importer patch si pas déjà fait, ou utiliser monkeypatch
import json

# Ajouter la racine du projet au PYTHONPATH pour trouver src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Importer l'application FastAPI
    from src.backend.main import app
    # Importer les modèles Pydantic depuis models.py
    from src.backend.app.models import PredictRequest, PredictResponse
    # Importer le type FaceShape depuis facial_analysis.py
    from src.backend.app.facial_analysis import FaceShape
    from typing import get_args
except ImportError as e:
    print(f"Erreur d'import dans test_api: {e}")
    # Laisser l'erreur être levée pour un diagnostic clair
    raise e 

# Créer un client de test pour l'application
client = TestClient(app)

# --- Données de test --- 

def generate_valid_landmarks(num_landmarks: int = 468) -> list:
    """Génère une liste de landmarks aléatoires valides."""
    return [[random.uniform(-500, 500), random.uniform(-500, 500), random.uniform(-500, 500)] 
            for _ in range(num_landmarks)]

# --- Tests Fonctionnels --- 

@pytest.mark.fonctionnel
def test_predict_success():
    """Teste un appel réussi à l'API /predict avec des données valides."""
    valid_data = {"landmarks": generate_valid_landmarks(468)}
    
    response = client.post("/predict", json=valid_data)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_shape" in data
    assert "recommended_glasses" in data
    
    valid_shapes = list(get_args(FaceShape))
    assert data["predicted_shape"] in valid_shapes
    
    assert isinstance(data["recommended_glasses"], list)
    if data["recommended_glasses"]:
        valid_glasses_ids = ["purple1", "classic_black", "modern_red", "aviator_gold", "round_tortoise"]
        for glass_id in data["recommended_glasses"]:
            assert glass_id in valid_glasses_ids

@pytest.mark.erreur
def test_predict_invalid_data_missing_field():
    """Teste l'appel à /predict avec un champ manquant."""
    invalid_data = {} 
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

@pytest.mark.erreur
def test_predict_invalid_data_wrong_type():
    """Teste l'appel à /predict avec un mauvais type pour landmarks."""
    invalid_data = {"landmarks": "ceci n'est pas une liste"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

@pytest.mark.erreur
def test_predict_invalid_data_wrong_landmark_count():
    """Teste l'appel à /predict avec un nombre incorrect de landmarks."""
    invalid_data_less = {"landmarks": generate_valid_landmarks(100)}
    response_less = client.post("/predict", json=invalid_data_less)
    assert response_less.status_code == 422
    
    invalid_data_more = {"landmarks": generate_valid_landmarks(500)}
    response_more = client.post("/predict", json=invalid_data_more)
    assert response_more.status_code == 422

@pytest.mark.erreur
def test_predict_invalid_data_wrong_landmark_format():
    """Teste l'appel à /predict avec un format incorrect pour un landmark."""
    landmarks = generate_valid_landmarks(467)
    landmarks.append([1.0, 2.0]) # Landmark invalide
    invalid_data = {"landmarks": landmarks}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

@pytest.mark.erreur
def test_predict_internal_error(monkeypatch):
    """Teste la gestion d'une erreur inattendue pendant la classification."""
    valid_data = {"landmarks": generate_valid_landmarks(468)}
    
    def mock_classify_error(*args, **kwargs):
        raise ValueError("Erreur simulée pendant l'estimation")

    # Cibler la fonction MLP réellement utilisée dans la route /predict de main.py
    monkeypatch.setattr("src.backend.main.estimate_face_shape_mlp", mock_classify_error)

    response = client.post("/predict", json=valid_data)
    
    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Erreur interne du serveur lors de la prédiction."

@pytest.mark.fonctionnel
def test_read_root():
    """Teste la route racine /."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API d'analyse faciale!"} 

# --- Test Fonctionnel (Endpoint /predict) ---

@pytest.mark.fonctionnel
def test_predict_endpoint_success_detailed():
    """Teste l'endpoint /predict avec des données de landmarks valides (simulées)."""
    mock_landmarks: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 468
    
    response = client.post("/predict", json={"landmarks": mock_landmarks})
    
    assert response.status_code == 200
    
    data = response.json()
    # Utiliser les noms de clés retournés par l'API (vérifiés dans test_predict_success)
    face_shape_key = "predicted_shape" 
    recommendations_key = "recommended_glasses"
    
    assert face_shape_key in data
    assert recommendations_key in data
    
    assert isinstance(data[face_shape_key], str)
    assert isinstance(data[recommendations_key], list)
    
    valid_shapes = ["Ovale", "Carrée", "Ronde", "Coeur", "Longue", "Inconnue"]
    assert data[face_shape_key] in valid_shapes
    
    if data[face_shape_key] != "Inconnue": 
        assert len(data[recommendations_key]) > 0 
        # Vérifier que les recommandations sont des chaînes (IDs)
        for rec in data[recommendations_key]:
             assert isinstance(rec, str)
             assert rec # Non vide
    else:
         assert len(data[recommendations_key]) == 0 

# --- Test d'Erreur (Endpoint /predict) ---

@pytest.mark.erreur
@pytest.mark.parametrize("invalid_payload", [
    {"landmarks": "ceci n'est pas une liste"}, 
    {"landmarks": [[0,0,0], [1,1]]},          
    {"landmarks": []},                         
    {"landmarks": [(0,0,0)] * 10},             
    {"autre_cle": [(0,0,0)] * 468},           
    {},                                         
])
def test_predict_endpoint_invalid_input_detailed(invalid_payload: dict):
    """Teste l'endpoint /predict avec différents types de données invalides."""
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422
    
    data = response.json()
    assert "detail" in data 
    # On pourrait vérifier plus spécifiquement le message d'erreur si nécessaire
    # print(data["detail"]) 

# --- Tests Fonctionnels API (/predict) --- 

def test_predict_valid_request():
    """Teste la route /predict avec une requête valide (landmarks simulés)."""
    valid_landmarks = [(0.0, 0.0, 0.0)] * 468 
    assert len(valid_landmarks) == 468
    request_data = {"landmarks": valid_landmarks}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200, f"Code de statut inattendu: {response.status_code}, Body: {response.text}"
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        pytest.fail(f"La réponse n'est pas un JSON valide: {response.text}")
    assert "predicted_shape" in response_data
    assert "recommended_glasses" in response_data
    assert isinstance(response_data["predicted_shape"], str)
    assert isinstance(response_data["recommended_glasses"], list)
    possible_shapes = list(get_args(FaceShape))
    assert response_data["predicted_shape"] in possible_shapes, \
        f"Forme prédite '{response_data['predicted_shape']}' non valide. Attendu une de {possible_shapes}"
    print(f"Test /predict OK - Statut: {response.status_code}, Forme: {response_data['predicted_shape']}")

# --- Tests d'Erreur API (/predict) --- 

def test_predict_invalid_landmarks_format():
    """Teste la route /predict avec un format de landmarks invalide (pas une liste)."""
    request_data = {"landmarks": "ceci n'est pas une liste"}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("Test /predict erreur (format landmarks invalide) OK - Statut 422")

def test_predict_wrong_number_of_landmarks():
    """Teste la route /predict avec un nombre incorrect de landmarks."""
    invalid_landmarks = [(random.random(), random.random(), random.random()) for _ in range(100)]
    request_data = {"landmarks": invalid_landmarks}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("Test /predict erreur (nombre landmarks incorrect) OK - Statut 422")

def test_predict_missing_landmarks_key():
    """Teste la route /predict sans la clé 'landmarks' dans le JSON."""
    request_data = {"autre_cle": "valeur"}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("Test /predict erreur (clé landmarks manquante) OK - Statut 422")

def test_predict_empty_payload():
    """Teste la route /predict avec un payload JSON vide."""
    request_data = {}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("Test /predict erreur (payload vide) OK - Statut 422")

def test_predict_non_json_payload():
    """Teste la route /predict avec un payload qui n'est pas du JSON."""
    response = client.post("/predict", content="ce n'est pas du json")
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("Test /predict erreur (payload non-JSON) OK - Statut 422") 