# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/routes/update.py

from flask import Blueprint, request, jsonify
from app.data_store import latest_data, update

# Create the blueprint
update_bp = Blueprint('update', __name__)

@update_bp.route('/update', methods=['POST'])
def update_data():
    """
    Update the shared latest_data dictionary with incoming JSON from Raspberry Pi.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Update shared latest_data
        update(data)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400