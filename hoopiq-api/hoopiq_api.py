# HoopIQ Cloud API
# Flask backend for receiving data from Raspberry Pi and serving to website

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import time
import base64

app = Flask(__name__)
CORS(app)

latest_data = {
    "basketball_fps": 0.0,
    "makes": 0,
    "attempts": 0,
    "swishes": 0,
    "backboard_hits": 0,
    "backboard_makes": 0,
    "backboard_misses": 0,
    "rim_hits": 0,
    "fg_percent": "0.0",
    "last_shot_type": "—",
    "trajectories": 0,
    "heatmap_fps": 0.0,
    "current_players": 0,
    "total_players_detected": 0,
    "heatmap_points": 0,
    "shot_chart": [],
    "basketball_frame": "",
    "heatmap_frame": "",
    "timestamp": 0,
    "last_update": 0
}

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "name": "HoopIQ API",
        "version": "2.0",
        "endpoints": {
            "/update": "POST - Pi sends data here",
            "/stats":  "GET  - Get current statistics",
            "/health": "GET  - Check if Pi is online"
        }
    })

@app.route('/update', methods=['POST'])
def update_data():
    global latest_data
    try:
        data = request.json

        latest_data['basketball_fps']        = data.get('basketball_fps', 0)
        latest_data['makes']                 = data.get('makes', 0)
        latest_data['attempts']              = data.get('attempts', 0)
        latest_data['swishes']               = data.get('swishes', 0)
        latest_data['backboard_hits']        = data.get('backboard_hits', 0)
        latest_data['backboard_makes']       = data.get('backboard_makes', 0)
        latest_data['backboard_misses']      = data.get('backboard_misses', 0)
        latest_data['rim_hits']              = data.get('rim_hits', 0)
        latest_data['fg_percent']            = data.get('fg_percent', '0.0')
        latest_data['last_shot_type']        = data.get('last_shot_type', '—')
        latest_data['trajectories']          = data.get('trajectories', 0)
        latest_data['heatmap_fps']           = data.get('heatmap_fps', 0)
        latest_data['current_players']       = data.get('current_players', 0)
        latest_data['total_players_detected']= data.get('total_players_detected', 0)
        latest_data['heatmap_points']        = data.get('heatmap_points', 0)
        latest_data['shot_chart']            = data.get('shot_chart', [])
        latest_data['basketball_frame']      = data.get('basketball_frame', '')
        latest_data['heatmap_frame']         = data.get('heatmap_frame', '')
        latest_data['timestamp']             = data.get('timestamp', 0)
        latest_data['last_update']           = time.time()

        return jsonify({"status": "success", "timestamp": latest_data['last_update']})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/stats', methods=['GET'])
def get_stats():
    is_online = (time.time() - latest_data['last_update']) < 5

    # Compute 2pt / 3pt breakdown from shot_chart
    two_makes = two_att = three_makes = three_att = 0
    three_zones = {"Left Corner 3", "Right Corner 3", "Above Break 3"}
    for shot in latest_data['shot_chart']:
        zone = shot.get('zone', '')
        made = shot.get('made', False)
        if zone in three_zones:
            three_att += 1
            if made: three_makes += 1
        else:
            two_att += 1
            if made: two_makes += 1

    two_pct   = round(two_makes   / two_att   * 100, 1) if two_att   > 0 else None
    three_pct = round(three_makes / three_att * 100, 1) if three_att > 0 else None

    # Compute current streak from shot_chart
    streak = 0
    for shot in reversed(latest_data['shot_chart']):
        if shot.get('made'):
            if streak >= 0: streak += 1
            else: break
        else:
            if streak <= 0: streak -= 1
            else: break

    return jsonify({
        "basketball": {
            "fps":              latest_data['basketball_fps'],
            "makes":            latest_data['makes'],
            "attempts":         latest_data['attempts'],
            "swishes":          latest_data['swishes'],
            "fg_percent":       latest_data['fg_percent'],
            "two_pt_percent":   two_pct,
            "three_pt_percent": three_pct,
            "last_shot_type":   latest_data['last_shot_type'],
            "trajectories":     latest_data['trajectories'],
            "streak":           streak,
            "shot_chart":       latest_data['shot_chart'],
            "avg_arc":          None,
            "avg_entry_angle":  None,
        },
        "heatmap": {
            "fps":             latest_data['heatmap_fps'],
            "current_players": latest_data['current_players'],
            "total_detected":  latest_data['total_players_detected'],
            "heatmap_points":  latest_data['heatmap_points'],
        },
        "sensors": {
            "backboard_hits":   latest_data['backboard_hits'],
            "backboard_makes":  latest_data['backboard_makes'],
            "backboard_misses": latest_data['backboard_misses'],
            "rim_hits":         latest_data['rim_hits'],
        },
        "system": {
            "online":      is_online,
            "last_update": latest_data['last_update'],
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    is_online = (time.time() - latest_data['last_update']) < 5
    return jsonify({
        "status":              "online" if is_online else "offline",
        "last_update":         latest_data['last_update'],
        "seconds_since_update": time.time() - latest_data['last_update']
    })

if __name__ == '__main__':
    print("HoopIQ Cloud API v2.0 Starting...")
    app.run(host='0.0.0.0', port=8000, debug=True)
