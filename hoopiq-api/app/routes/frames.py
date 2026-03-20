# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/routes/frames.py

from flask import Blueprint, Response
from app.data_store import latest_data
import base64

# Create the blueprint
frames_bp = Blueprint('frames', __name__)

@frames_bp.route('/frame/basketball', methods=['GET'])
def get_basketball_frame():
    """
    Return the latest basketball frame as a JPEG image.
    """
    try:
        frame_b64 = latest_data['basketball_frame']
        if not frame_b64:
            return "No frame available", 404
        return Response(base64.b64decode(frame_b64), mimetype='image/jpeg')
    except Exception as e:
        return f"Error: {str(e)}", 500

@frames_bp.route('/frame/heatmap', methods=['GET'])
def get_heatmap_frame():
    """
    Return the latest heatmap frame as a JPEG image.
    """
    try:
        frame_b64 = latest_data['heatmap_frame']
        if not frame_b64:
            return "No frame available", 404
        return Response(base64.b64decode(frame_b64), mimetype='image/jpeg')
    except Exception as e:
        return f"Error: {str(e)}", 500