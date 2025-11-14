# ball_tracker.py
# Real-time basketball tracking using OpenCV
# Runs on Raspberry Pi

import cv2
import numpy as np
import json
import time

# HSV color range for orange basketball
LOWER_ORANGE = np.array([5, 100, 100])
UPPER_ORANGE = np.array([15, 255, 255])

def track_ball(video_source=0):
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be ball)
            largest = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest) > 500:  # Min size filter
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Send coordinates to backend
                    shot_data = {
                        "x": cx,
                        "y": cy,
                        "timestamp": time.time()
                    }
                    print(json.dumps(shot_data))
                    
                    # Visual feedback
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        
        cv2.imshow('Ball Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_ball()
