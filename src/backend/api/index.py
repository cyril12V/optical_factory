from http.server import BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "status": "ok",
            "message": "API Optical Factory fonctionnelle",
            "endpoints": [
                {
                    "path": "/api/facial_analysis",
                    "method": "POST",
                    "description": "Analyse faciale pour recommandation de lunettes"
                }
            ]
        }
        
        self.wfile.write(json.dumps(response).encode('utf-8'))
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers() 