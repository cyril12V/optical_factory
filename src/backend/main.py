from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

# Imports relatifs au dossier courant
from app.models import PredictRequest, PredictResponse, LandmarkModel
from app.facial_analysis import (
    estimate_face_shape_mlp,
    get_recommendations,
    estimate_face_shape_from_landmarks_heuristic,
    flatten_landmarks_for_prediction,
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer l'instance de l'application FastAPI
app = FastAPI(
    title="Optical Factory API",
    description="API pour l'analyse de forme de visage et recommandation de lunettes utilisant un modèle MLP.",
    version="0.1.1" # Version pour refléter le retour au MLP
)

# Configuration CORS
# TODO: Restreindre les origines en production !
origins = [
    "http://localhost:3000",  # Adresse typique de React en développement
    "localhost:3000", # Parfois nécessaire sans http://
    "*" # Autorise toutes les origines pour le déploiement Vercel
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Autoriser toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"], # Autoriser tous les headers
)

# Route simple pour vérifier que le serveur fonctionne
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Bienvenue sur l'API d'analyse faciale!"}

# Route pour la prédiction de forme et la recommandation
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest = Body(...)):
    """
    Reçoit 468 landmarks faciaux et retourne la forme de visage prédite par le modèle MLP
    ainsi que les recommandations de lunettes associées.
    (Peut utiliser une heuristique comme fallback).
    """
    try:
        logger.info(f"Requête reçue sur /predict avec {len(request.landmarks)} landmarks.")

        # Vérification simple du nombre de landmarks
        if len(request.landmarks) != 468:
            logger.warning(f"Nombre de landmarks invalide: {len(request.landmarks)}")
            raise HTTPException(status_code=400,
                                detail=f"Nombre de landmarks invalide. Attendu 468, reçu {len(request.landmarks)}.")

        # Appel de la fonction de prédiction MLP
        landmarks_array = flatten_landmarks_for_prediction(list(request.landmarks))
        predicted_shape = estimate_face_shape_mlp(landmarks_array)
        logger.info(f"Forme prédite par MLP: {predicted_shape}")

        # Gestion si le modèle retourne une erreur interne ou "Inconnue"
        if "Erreur:" in predicted_shape:
             logger.error("Erreur interne retournée par le modèle MLP.")
             # Optionnel: Fallback vers heuristique ou retourner erreur 500
             predicted_shape = "Inconnue" # Ou lever HTTPException
             # raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction MLP.")
        
        # Optionnel: Fallback vers heuristique si MLP retourne "Inconnue"
        if predicted_shape == "Inconnue":
            logger.info("MLP a retourné 'Inconnue', tentative avec l'heuristique...")
            predicted_shape = estimate_face_shape_from_landmarks_heuristic(request.landmarks)
            logger.info(f"Forme prédite par heuristique: {predicted_shape}")

        # Récupérer les recommandations pour la forme prédite (même si "Inconnue")
        recommendations = get_recommendations(predicted_shape)
        logger.info(f"Recommandations pour {predicted_shape}: {recommendations}")

        return PredictResponse(predicted_shape=predicted_shape, recommended_glasses=recommendations)

    except HTTPException as http_exc: # Laisser passer les erreurs HTTP déjà levées
        raise http_exc
    except Exception as e:
        logger.exception(f"Erreur inattendue lors du traitement de /predict: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur lors de la prédiction.")

# Pour exécuter le serveur localement (par exemple avec: uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    # Attention: --reload est utile pour le dev, mais consomme plus de ressources
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 