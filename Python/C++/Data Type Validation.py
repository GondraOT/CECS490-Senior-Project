# data_validator.py
# Validates sensor data format before sending to Chris's web app

from typing import TypedDict, Optional
import json

class ShotData(TypedDict):
    x: float  # 0-100 (normalized court coordinates)
    y: float  # 0-100
    make: bool
    timestamp: str  # ISO 8601 format

def validate_shot_data(data: dict) -> Optional[ShotData]:
    """
    Validates incoming sensor data
    Returns validated data or None if invalid
    """
    try:
        # Type checking
        x = float(data.get('x', -1))
        y = float(data.get('y', -1))
        make = bool(data.get('make', False))
        timestamp = str(data.get('timestamp', ''))
        
        # Range validation
        if not (0 <= x <= 100 and 0 <= y <= 100):
            print(f"ERROR: Invalid coordinates x={x}, y={y}")
            return None
        
        if not timestamp:
            print("ERROR: Missing timestamp")
            return None
        
        return {
            'x': x,
            'y': y,
            'make': make,
            'timestamp': timestamp
        }
    
    except (ValueError, TypeError) as e:
        print(f"ERROR: Data validation failed - {e}")
        return None

# Test with sample data
if __name__ == "__main__":
    test_data = {
        "x": 45.3,
        "y": 78.2,
        "make": True,
        "timestamp": "2025-10-15T14:30:00Z"
    }
    
    validated = validate_shot_data(test_data)
    if validated:
        print("✓ Valid:", json.dumps(validated, indent=2))
