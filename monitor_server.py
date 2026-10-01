import http.server
import socketserver
import json
import os

PORT = 8080
LOG_FILE = "full_pipeline_run.log"

class MonitorHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/logs':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                if not os.path.exists(LOG_FILE):
                    content = "Log file not found."
                else:
                    # Read last 100 lines
                    with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                        last_lines = lines[-100:]
                        content = "".join(last_lines)
            except Exception as e:
                content = f"Error reading log: {str(e)}"
                
            response = json.dumps({"logs": content})
            self.wfile.write(response.encode())
        else:
            # Default behavior (serves index.html for root)
            super().do_GET()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), MonitorHandler) as httpd:
        print(f"Monitor Server running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
