from pydantic import BaseModel, Field
from typing import List, Tuple

# Type pour un seul point de repère (landmark)
# Doit correspondre à la structure envoyée par le frontend
LandmarkModel = Tuple[float, float, float]

# Schéma pour les données d'entrée de la requête /predict
class PredictRequest(BaseModel):
    landmarks: List[LandmarkModel] = Field(..., 
                                            description="Liste des 468 points de repère du visage de Facemesh", 
                                            min_items=468, 
                                            max_items=468)

# Schéma pour la réponse de la requête /predict
class PredictResponse(BaseModel):
    predicted_shape: str = Field(..., description="Forme du visage prédite par le modèle")
    recommended_glasses: List[str] = Field(..., description="Liste des IDs des lunettes recommandées") 