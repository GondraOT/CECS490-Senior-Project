# HoopIQ Cloud API
# Flask backend for receiving data from Raspberry Pi and serving to website

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import time
import base64

app = Flask(__name__)
CORS(app)  # Allow requests from any domain

# Store latest data in memory
latest_data = {
    "basketball_fps": 0.0,
    "makes": 0,
    "trajectories": 0,
    "backboard_hits": 0,
    "rim_hits": 0,
    "heatmap_fps": 0.0,
    "current_players": 0,
    "total_players_detected": 0,
    "heatmap_points": 0,
    "basketball_frame": "",
    "heatmap_frame": "",
    "timestamp": 0,
    "last_update": 0
}

@app.route('/', methods=['GET'])
def home():
    """API info page"""
    return jsonify({
        "name": "HoopIQ API",
        "version": "1.0",
        "endpoints": {
            "/update": "POST - Pi sends data here",
            "/stats": "GET - Get current statistics",
            "/frame/basketball": "GET - Get latest basketball frame",
            "/frame/heatmap": "GET - Get latest heatmap frame",
            "/health": "GET - Check if Pi is online"
        }
    })

@app.route('/update', methods=['POST'])
def update_data():
    """Raspberry Pi posts data here every second"""
    global latest_data
    
    try:
        data = request.json
        
        # Update all fields
        latest_data['basketball_fps'] = data.get('basketball_fps', 0)
        latest_data['makes'] = data.get('makes', 0)
        latest_data['trajectories'] = data.get('trajectories', 0)
        latest_data['backboard_hits'] = data.get('backboard_hits', 0)
        latest_data['rim_hits'] = data.get('rim_hits', 0)
        latest_data['heatmap_fps'] = data.get('heatmap_fps', 0)
        latest_data['current_players'] = data.get('current_players', 0)
        latest_data['total_players_detected'] = data.get('total_players_detected', 0)
        latest_data['heatmap_points'] = data.get('heatmap_points', 0)
        latest_data['basketball_frame'] = data.get('basketball_frame', '')
        latest_data['heatmap_frame'] = data.get('heatmap_frame', '')
        latest_data['timestamp'] = data.get('timestamp', 0)
        latest_data['last_update'] = time.time()
        
        return jsonify({"status": "success", "timestamp": latest_data['last_update']})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get current statistics - used by website"""
    is_online = (time.time() - latest_data['last_update']) < 5  # Online if updated in last 5 seconds
    
    return jsonify({
        "basketball": {
            "fps": latest_data['basketball_fps'],
            "makes": latest_data['makes'],
            "trajectories": latest_data['trajectories']
        },
        "heatmap": {
            "fps": latest_data['heatmap_fps'],
            "current_players": latest_data['current_players'],
            "total_detected": latest_data['total_players_detected'],
            "heatmap_points": latest_data['heatmap_points']
        },
        "sensors": {
            "backboard_hits": latest_data['backboard_hits'],
            "rim_hits": latest_data['rim_hits']
        },
        "system": {
            "online": is_online,
            "last_update": latest_data['last_update'],
            "uptime_seconds": time.time() - latest_data['last_update'] if is_online else 0
        }
    })

@app.route('/stream/basketball', methods=['GET'])
def stream_basketball():
    """Stream basketball frames as MJPEG"""
    def generate():
        while True:
            if latest_data['basketball_frame']:
                try:
                    frame_data = base64.b64decode(latest_data['basketball_frame'])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                except Exception as e:
                    print(f"Stream error: {e}")
                    pass
            time.sleep(0.033)  # 30 FPS (1/30 = 0.033 seconds)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/heatmap', methods=['GET'])
def stream_heatmap():
    """Stream heatmap frames as MJPEG"""
    def generate():
        while True:
            if latest_data['heatmap_frame']:
                try:
                    frame_data = base64.b64decode(latest_data['heatmap_frame'])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                except Exception as e:
                    print(f"Stream error: {e}")
                    pass
            time.sleep(0.016)  # 60 FPS (1/60 = 0.016 seconds)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health', methods=['GET'])
def health_check():
    """Check if system is healthy"""
    is_online = (time.time() - latest_data['last_update']) < 5
    
    return jsonify({
        "status": "online" if is_online else "offline",
        "last_update": latest_data['last_update'],
        "seconds_since_update": time.time() - latest_data['last_update']
    })

if __name__ == '__main__':
    print("HoopIQ Cloud API Starting...")
    print("Endpoints:")
    print("  POST /update - Receive data from Pi")
    print("  GET  /stats - Get statistics")
    print("  GET  /frame/basketball - Get basketball frame")
    print("  GET  /frame/heatmap - Get heatmap frame")
    print("  GET  /health - System health check")
    app.run(host='0.0.0.0', port=8000, debug=True)
