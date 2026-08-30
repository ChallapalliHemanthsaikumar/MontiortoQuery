"""Simple dashboard server that serves captured data."""

import http.server
import json
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PORT = 5000
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Serve dashboard files and API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

    def do_GET(self):
        if self.path == '/data/metrics.json':
            self._serve_json('test/metrics.json')
        elif self.path == '/data/events.json':
            self._serve_json('test/events.json')
        elif self.path.startswith('/data/captures/'):
            self._serve_captures()
        else:
            super().do_GET()

    def _serve_json(self, filename):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def _serve_captures(self):
        if self.path == '/data/captures/':
            captures_dir = os.path.join(DATA_DIR, 'captures', 'exp001')
            if os.path.exists(captures_dir):
                files = sorted([f for f in os.listdir(captures_dir) if f.endswith('.jpg')])
                links = [f'/data/captures/{f}' for f in files]
                html = '\n'.join(f'<a href="{l}">{l}</a>' for l in links)
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(b'')
        else:
            filename = self.path.split('/')[-1]
            filepath = os.path.join(DATA_DIR, 'captures', 'exp001', filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)


def main():
    print(f"Starting AntVision Dashboard on http://localhost:{PORT}")
    print(f"Serving data from: {DATA_DIR}")
    server = http.server.HTTPServer(('0.0.0.0', PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
