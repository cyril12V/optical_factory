from http.server import BaseHTTPRequestHandler
import json
import sys
import os
import numpy as np

# Ajout du répertoire parent pour les imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Imports relatifs au projet
try:
    from app.facial_analysis import flatten_landmarks_for_prediction, estimate_face_shape_mlp, get_recommendations
except ImportError:
    # Fallback pour l'environnement Vercel
    def flatten_landmarks_for_prediction(landmarks_list):
        if len(landmarks_list) != 468:
            raise ValueError("Nombre de landmarks incorrect")
        return np.array(landmarks_list).flatten().reshape(1, -1)
    
    def estimate_face_shape_mlp(landmarks_flat):
        return "Ovale"  # Valeur par défaut si l'import échoue
    
    def get_recommendations(shape):
        recommandations = {
            "Ovale": ["Lunettes rectangulaires", "Aviator", "Wayfarer"],
            "Ronde": ["Lunettes rectangulaires", "Carrées", "Angulaires"],
            "Carrée": ["Lunettes rondes", "Ovales", "Aviator"],
            "Coeur": ["Lunettes rondes", "Aviator", "Cat-eye"],
            "Longue": ["Lunettes rondes", "Carrées", "Wraparound"]
        }
        return recommandations.get(shape, ["Modèle universel"])

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        try:
            # Récupérer les points du visage
            facial_points = data.get('facial_points', [])
            
            # Préparer les données pour la prédiction
            landmarks_flat = flatten_landmarks_for_prediction(facial_points)
            
            # Estimer la forme du visage
            face_shape = estimate_face_shape_mlp(landmarks_flat)
            
            # Obtenir les recommandations
            recommendations = get_recommendations(face_shape)
            
            # Préparer la réponse
            response_data = {
                "face_shape": face_shape,
                "recommendations": recommendations
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "message": "Erreur lors de l'analyse faciale"
            }
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
            
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers() 