# HoopIQ Cloud API
# Flask backend for receiving data from Raspberry Pi and serving to website

from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
import time
import base64
import os

app = Flask(__name__)
CORS(app)

latest_data = {
    "basketball_fps": 0.0,
    "makes": 0,
    "attempts": 0,
    "trajectories": 0,
    "backboard_hits": 0,
    "rim_hits": 0,
    "swishes": 0,
    "streak": 0,
    "avg_arc": None,
    "avg_entry_angle": None,
    "shot_chart": [],
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
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/update', methods=['POST'])
def update_data():
    global latest_data
    try:
        data = request.json
        latest_data['basketball_fps']        = data.get('basketball_fps', 0)
        latest_data['makes']                 = data.get('makes', 0)
        latest_data['attempts']              = data.get('attempts', 0)
        latest_data['trajectories']          = data.get('trajectories', 0)
        latest_data['backboard_hits']        = data.get('backboard_hits', 0)
        latest_data['rim_hits']              = data.get('rim_hits', 0)
        latest_data['swishes']               = data.get('swishes', 0)
        latest_data['streak']                = data.get('streak', 0)
        latest_data['avg_arc']               = data.get('avg_arc', None)
        latest_data['avg_entry_angle']       = data.get('avg_entry_angle', None)
        latest_data['shot_chart']            = data.get('shot_chart', [])
        latest_data['heatmap_fps']           = data.get('heatmap_fps', 0)
        latest_data['current_players']       = data.get('current_players', 0)
        latest_data['total_players_detected']= data.get('total_players_detected', 0)
        latest_data['heatmap_points']        = data.get('heatmap_points', 0)
        latest_data['basketball_frame']      = data.get('basketball_frame', '')
        latest_data['heatmap_frame']         = data.get('heatmap_frame', '')
        latest_data['timestamp']             = data.get('timestamp', 0)
        latest_data['last_update']           = time.time()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/stats', methods=['GET'])
def get_stats():
    is_online = (time.time() - latest_data['last_update']) < 5
    makes    = latest_data['makes']
    attempts = latest_data['attempts']

    # Calculate percentages — Flask handles this so Rust doesn't need to
    fg_pct = round((makes / attempts * 100), 1) if attempts > 0 else None

    # Shot chart zone breakdown for 2pt/3pt
    shot_chart = latest_data['shot_chart']
    two_makes = sum(1 for s in shot_chart if s.get('zone') == '2pt' and s.get('made'))
    two_att   = sum(1 for s in shot_chart if s.get('zone') == '2pt')
    three_makes = sum(1 for s in shot_chart if s.get('zone') == '3pt' and s.get('made'))
    three_att   = sum(1 for s in shot_chart if s.get('zone') == '3pt')

    two_pct   = round((two_makes / two_att * 100), 1)   if two_att   > 0 else None
    three_pct = round((three_makes / three_att * 100), 1) if three_att > 0 else None

    return jsonify({
        "basketball": {
            "fps":        latest_data['basketball_fps'],
            "makes":      makes,
            "attempts":   attempts,
            "trajectories": latest_data['trajectories'],
            "fg_percent": fg_pct,
            "two_pt_percent":   two_pct,
            "three_pt_percent": three_pct,
            "swishes":    latest_data['swishes'],
            "streak":     latest_data['streak'],
            "avg_arc":    latest_data['avg_arc'],
            "avg_entry_angle": latest_data['avg_entry_angle'],
            "shot_chart": shot_chart,
        },
        "heatmap": {
            "fps":             latest_data['heatmap_fps'],
            "current_players": latest_data['current_players'],
            "total_detected":  latest_data['total_players_detected'],
            "heatmap_points":  latest_data['heatmap_points'],
        },
        "sensors": {
            "backboard_hits": latest_data['backboard_hits'],
            "rim_hits":       latest_data['rim_hits'],
        },
        "system": {
            "online":      is_online,
            "last_update": latest_data['last_update'],
        }
    })

@app.route('/frame/basketball', methods=['GET'])
def get_basketball_frame():
    try:
        if not latest_data['basketball_frame']:
            return "No frame available", 404
        return Response(base64.b64decode(latest_data['basketball_frame']), mimetype='image/jpeg')
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/frame/heatmap', methods=['GET'])
def get_heatmap_frame():
    try:
        if not latest_data['heatmap_frame']:
            return "No frame available", 404
        return Response(base64.b64decode(latest_data['heatmap_frame']), mimetype='image/jpeg')
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/health', methods=['GET'])
def health_check():
    is_online = (time.time() - latest_data['last_update']) < 5
    return jsonify({
        "status": "online" if is_online else "offline",
        "last_update": latest_data['last_update'],
        "seconds_since_update": time.time() - latest_data['last_update']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
