# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/routes/reset.py

from flask import Blueprint, jsonify
from app.data_store import latest_data, update

reset_bp = Blueprint('reset', __name__)

@reset_bp.route('/reset', methods=['POST'])
def reset_data():
    """
    Reset the shared latest_data dictionary to default values.
    """
    reset_values = {
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
        "heatmap_points": 0,
        "total_players_detected": 0
    }
    update(reset_values)
    return jsonify({"status": "reset"})