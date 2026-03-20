# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/data_store.py

import time

# Shared dictionary for all routes
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

def update(new_data: dict):
    """
    Safely update the shared latest_data dictionary.
    Only updates keys that exist in latest_data.
    Automatically updates 'last_update' timestamp.
    """
    for key, value in new_data.items():
        if key in latest_data:
            latest_data[key] = value
    latest_data['last_update'] = time.time()