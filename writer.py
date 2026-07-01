import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get frame dimensions
ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
else:
    h, w = 480, 640

# Create a blank canvas
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Drawing state
drawing_enabled = False      # Toggled by 'W'
symmetry_enabled = False     # Toggled by 'M'
prev_x, prev_y = 0, 0

# Circle tool state
circle_mode = False
circle_center = (0, 0)
circle_radius = 50
circle_timer_start = None
PALM_CLOSE_DURATION = 2.0    # seconds
MIN_RADIUS, MAX_RADIUS = 20, 200

def is_pointing(hand_landmarks):
    """True if index finger is extended toward camera (tip closer than MCP)."""
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return tip.z < mcp.z

def is_palm_close(hand_landmarks):
    """
    Detect if hand is open (all fingers extended) and close to camera.
    Returns True if all fingertips are above their PIP joints and average z < -0.1.
    """
    tips = [4, 8, 12, 16, 20]       # Thumb tip, index tip, etc.
    pips = [3, 6, 10, 14, 18]       # Corresponding PIP joints

    all_extended = True
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            all_extended = False
            break

    # Average z of all landmarks
    avg_z = np.mean([lm.z for lm in hand_landmarks.landmark])
    close_to_camera = avg_z < -0.1   # negative is closer

    return all_extended and close_to_camera

def get_palm_center(hand_landmarks):
    """Average of wrist, index MCP, pinky MCP, and middle MCP for a stable center."""
    idx = [0, 5, 9, 13, 17]          # wrist and base of each finger
    xs = [hand_landmarks.landmark[i].x for i in idx]
    ys = [hand_landmarks.landmark[i].y for i in idx]
    center_x = int(np.mean(xs) * w)
    center_y = int(np.mean(ys) * h)
    return (center_x, center_y)

def pinch_distance(hand_landmarks):
    """Distance between thumb tip (4) and index tip (8) in pixels."""
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    dist = np.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
    return dist

print("AirWriter Enhanced. Controls:")
print("  W - toggle drawing on/off")
print("  M - toggle symmetry on/off")
print("  C - clear canvas (or fix circle when in circle mode)")
print("  Q - quit")
print("\nTo draw: press W, then point index finger at camera and move.")
print("To place a circle: hold open palm close to camera for 2s.")
print("  - Move your hand to position the circle.")
print("  - Pinch thumb and index to resize.")
print("  - Press C to fix the circle (outline) on the canvas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror view (do NOT flip canvas)
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Default overlay shows only the camera feed
    overlay = frame.copy()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Detect palm close gesture (used both for timer and to inhibit drawing)
        palm_close = is_palm_close(hand_landmarks)

        # ----- Circle Mode Handling -----
        if circle_mode:
            # Circle follows the palm in real time
            circle_center = get_palm_center(hand_landmarks)
            # Pinch to resize
            dist = pinch_distance(hand_landmarks)
            circle_radius = int(np.clip(dist * 0.8, MIN_RADIUS, MAX_RADIUS))
            # Draw the moving circle on overlay
            cv2.circle(overlay, circle_center, circle_radius, (0, 255, 255), 3)
            cv2.putText(overlay, "Move to position, pinch to resize, press C to fix", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # ----- Palm Close Timer for Circle Activation -----
            if palm_close:
                if circle_timer_start is None:
                    circle_timer_start = time.time()
                elif time.time() - circle_timer_start >= PALM_CLOSE_DURATION:
                    # Enter circle mode
                    circle_mode = True
                    circle_center = get_palm_center(hand_landmarks)
                    circle_radius = 50   # default
                    circle_timer_start = None
                    # Disable drawing while in circle mode
                    drawing_enabled = False
            else:
                circle_timer_start = None

            # ----- Drawing (only if enabled, pointing, palm NOT close, and NOT in circle mode) -----
            if drawing_enabled and is_pointing(hand_landmarks) and not palm_close and not circle_mode:
                tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                tip_x, tip_y = int(tip.x * w), int(tip.y * h)

                if prev_x != 0 and prev_y != 0:
                    # Draw original line
                    cv2.line(canvas, (prev_x, prev_y), (tip_x, tip_y), (0, 255, 0), 8)
                    # If symmetry is on, draw mirrored line
                    if symmetry_enabled:
                        cv2.line(canvas, (w - prev_x, prev_y), (w - tip_x, tip_y), (0, 255, 0), 8)

                prev_x, prev_y = tip_x, tip_y
            else:
                # Reset line continuity when not drawing
                prev_x, prev_y = 0, 0

    # Combine frame and canvas (permanent drawings) with transparency
    overlay = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)

    # Display current modes on screen
    mode_text = f"Draw: {'ON' if drawing_enabled else 'OFF'}  Symmetry: {'ON' if symmetry_enabled else 'OFF'}"
    cv2.putText(overlay, mode_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("AirWriter", overlay)

    # Keyboard handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if circle_mode:
            # Fix the circle permanently on canvas (outline) and exit circle mode
            cv2.circle(canvas, circle_center, circle_radius, (0, 255, 255), 3)
            circle_mode = False
        else:
            # Clear entire canvas
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('w'):
        # Toggle drawing mode (automatically ignored when circle_mode is active)
        drawing_enabled = not drawing_enabled
        if not drawing_enabled:
            prev_x, prev_y = 0, 0
    elif key == ord('m'):
        symmetry_enabled = not symmetry_enabled

cap.release()
cv2.destroyAllWindows()