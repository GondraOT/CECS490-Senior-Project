# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/routes/health.py

from flask import Blueprint, jsonify
from app.data_store import latest_data
from app.utils import is_system_online
import time

# Create the blueprint
health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Return the system health status based on last update timestamp.
    """
    last_update = latest_data.get('last_update', 0)
    seconds_since_update = time.time() - last_update
    is_online = is_system_online(latest_data['last_update'])

    return jsonify({
        "status": "online" if is_online else "offline",
        "last_update": last_update,
        "seconds_since_update": round(seconds_since_update, 2)
    })