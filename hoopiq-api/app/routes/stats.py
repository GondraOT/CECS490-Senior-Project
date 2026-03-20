# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/routes/stats.py

from flask import Blueprint, jsonify
from app.data_store import latest_data
from app.utils import calculate_fg_percent, calculate_zone_percent, is_system_online
import time

# Create the blueprint
stats_bp = Blueprint('stats', __name__)

@stats_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Return basketball, heatmap, sensor, and system stats based on latest_data.
    """
    is_online = is_system_online(latest_data['last_update'])
    makes = latest_data['makes']
    attempts = latest_data['attempts']

    # Calculate overall field goal %
    fg_pct = calculate_fg_percent(latest_data['makes'], latest_data['attempts'])

    # Shot chart 2pt/3pt breakdown
    shot_chart = latest_data['shot_chart']
    two_pct = calculate_zone_percent(latest_data['shot_chart'], '2pt')
    three_pct = calculate_zone_percent(latest_data['shot_chart'], '3pt')

    return jsonify({
        "basketball": {
            "fps": latest_data['basketball_fps'],
            "makes": makes,
            "attempts": attempts,
            "trajectories": latest_data['trajectories'],
            "fg_percent": fg_pct,
            "two_pt_percent": two_pct,
            "three_pt_percent": three_pct,
            "swishes": latest_data['swishes'],
            "streak": latest_data['streak'],
            "avg_arc": latest_data['avg_arc'],
            "avg_entry_angle": latest_data['avg_entry_angle'],
            "shot_chart": shot_chart,
        },
        "heatmap": {
            "fps": latest_data['heatmap_fps'],
            "current_players": latest_data['current_players'],
            "total_detected": latest_data['total_players_detected'],
            "heatmap_points": latest_data['heatmap_points'],
        },
        "sensors": {
            "backboard_hits": latest_data['backboard_hits'],
            "rim_hits": latest_data['rim_hits'],
        },
        "system": {
            "online": is_online,
            "last_update": latest_data['last_update'],
        }
    })