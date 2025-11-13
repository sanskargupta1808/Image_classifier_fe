from http.server import BaseHTTPRequestHandler
import json
import sys
import os
sys.path.append('../src')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            from src.detect import AIImageDetector
            detector = AIImageDetector('../models')
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Process image data here
            result = {"prediction": 0, "confidence": 0.95}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
