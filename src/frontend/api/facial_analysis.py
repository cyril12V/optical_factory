from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Ajouter le chemin du backend au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.app.facial_analysis import analyze_face

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        # Analyse faciale
        result = analyze_face(data.get('facial_points', []))
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = json.dumps({"result": result})
        self.wfile.write(response.encode('utf-8'))
        return 