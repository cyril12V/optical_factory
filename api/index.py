from http.server import BaseHTTPRequestHandler
import sys
import os

# Ajouter le chemin du backend au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))

# Importer l'app FastAPI
from src.backend.main import app

# Créer un handler HTTP compatible avec Vercel
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(f'API Optical Factory est active! Accédez à /docs pour la documentation.'.encode())
        return 