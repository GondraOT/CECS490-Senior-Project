# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/utils.py

import time

def calculate_fg_percent(makes: int, attempts: int) -> float | None:
    """
    Calculate overall field goal percentage.
    Returns None if attempts is 0.
    """
    if attempts <= 0:
        return None
    return round((makes / attempts) * 100, 1)


def calculate_zone_percent(shot_chart: list, zone: str) -> float | None:
    """
    Calculate made percentage for a specific zone (e.g., '2pt', '3pt').
    Returns None if no attempts in that zone.

    Example:
        shot_chart = [
        {"zone": "2pt", "made": True},
        {"zone": "3pt", "made": False},
        ...
    ]
    """
    zone_makes = sum(1 for s in shot_chart if s.get('zone') == zone and s.get('made'))
    zone_att = sum(1 for s in shot_chart if s.get('zone') == zone)
    if zone_att == 0:
        return None
    return round((zone_makes / zone_att) * 100, 1)


def is_system_online(last_update: float, threshold: float = 5.0) -> bool:
    """
    Determine if the system is online based on last_update timestamp.
    """
    return (time.time() - last_update) < threshold