"""
Virtual Mouse Using Hand Gestures
Compatible with mediapipe 0.10.32 (Tasks API)
"""

import cv2
import numpy as np
import pyautogui
import time
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark

# ─────────────────────────────────────────────
# DOWNLOAD MODEL IF NEEDED
# ─────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~6MB)... please wait")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CAMERA_INDEX      = 0
FRAME_REDUCTION   = 100
SMOOTHENING       = 7
CLICK_THRESHOLD   = 0.05   # normalized distance (0.0 - 1.0)
SCROLL_SPEED      = 20
DOUBLE_CLICK_TIME = 0.3

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_x, prev_y  = 0, 0
curr_x, curr_y  = 0, 0
last_click_time = 0
clicking        = False
scroll_prev_y   = None

# Store latest detection result
latest_result   = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# ─────────────────────────────────────────────
# BUILD DETECTOR
# ─────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_result
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
# DRAW HAND CONNECTIONS
# ─────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def draw_hand(frame, lm_px):
    for a, b in CONNECTIONS:
        cv2.line(frame, lm_px[a], lm_px[b], (0, 200, 200), 2)
    for pt in lm_px:
        cv2.circle(frame, pt, 4, (255, 255, 255), cv2.FILLED)
        cv2.circle(frame, pt, 4, (0, 150, 255), 1)

# ─────────────────────────────────────────────
# FINGER STATE
# ─────────────────────────────────────────────
def fingers_up(lm, handedness="Right"):
    """Returns [thumb, index, middle, ring, pinky] as booleans."""
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    up = []
    # Thumb (x-axis)
    if handedness == "Right":
        up.append(lm[tips[0]].x < lm[joints[0]].x)
    else:
        up.append(lm[tips[0]].x > lm[joints[0]].x)
    # Other fingers (y-axis)
    for i in range(1, 5):
        up.append(lm[tips[i]].y < lm[joints[i]].y)
    return up

def norm_dist(p1, p2):
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("Virtual Mouse started! Show your hand to the camera.")
print("Press Q to quit.\n")

timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error. Try changing CAMERA_INDEX to 1.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Draw control zone
    cv2.rectangle(frame,
                  (FRAME_REDUCTION, FRAME_REDUCTION),
                  (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
                  (0, 200, 255), 2)

    # Send frame to detector
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp += 1
    detector.detect_async(mp_image, timestamp)

    if latest_result and latest_result.hand_landmarks:
        lm         = latest_result.hand_landmarks[0]
        handedness = latest_result.handedness[0][0].category_name

        # Convert normalized landmarks to pixel coords
        lm_px = [(int(p.x * w), int(p.y * h)) for p in lm]
        draw_hand(frame, lm_px)

        fingers = fingers_up(lm, handedness)

        index_tip  = lm[8]
        middle_tip = lm[12]
        thumb_tip  = lm[4]
        index_px   = lm_px[8]
        middle_px  = lm_px[12]

        # ── MOVE (only index up) ──
        if fingers[1] and not fingers[2]:
            x = np.interp(index_tip.x, 
                          [FRAME_REDUCTION/w, 1 - FRAME_REDUCTION/w],
                          [0, screen_w])
            y = np.interp(index_tip.y,
                          [FRAME_REDUCTION/h, 1 - FRAME_REDUCTION/h],
                          [0, screen_h])
            curr_x = prev_x + (x - prev_x) / SMOOTHENING
            curr_y = prev_y + (y - prev_y) / SMOOTHENING
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            clicking = False
            scroll_prev_y = None
            cv2.circle(frame, index_px, 12, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "MOVE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ── LEFT CLICK (index + middle, bring together) ──
        elif fingers[1] and fingers[2] and not fingers[3]:
            d = norm_dist(index_tip, middle_tip)
            cv2.putText(frame, "CLICK MODE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            cv2.line(frame, index_px, middle_px, (255, 165, 0), 2)
            if d < CLICK_THRESHOLD:
                now = time.time()
                if not clicking:
                    if now - last_click_time < DOUBLE_CLICK_TIME:
                        pyautogui.doubleClick()
                        cv2.putText(frame, "DOUBLE CLICK!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        pyautogui.click()
                        cv2.putText(frame, "CLICK!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    clicking = True
                    last_click_time = now
                cv2.circle(frame, index_px,  12, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, middle_px, 12, (0, 255, 255), cv2.FILLED)
            else:
                clicking = False
            scroll_prev_y = None

        # ── RIGHT CLICK (thumb + index pinch) ──
        elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
            d = norm_dist(thumb_tip, index_tip)
            cv2.putText(frame, "RIGHT CLICK MODE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 255), 2)
            if d < CLICK_THRESHOLD:
                pyautogui.rightClick()
                cv2.putText(frame, "RIGHT CLICK!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 255), 2)
                time.sleep(0.4)

        # ── SCROLL (index + middle + ring up) ──
        elif fingers[1] and fingers[2] and fingers[3]:
            cy = index_tip.y
            cv2.putText(frame, "SCROLL MODE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if scroll_prev_y is not None:
                delta = scroll_prev_y - cy
                if abs(delta) > 0.01:
                    pyautogui.scroll(int(delta * 500 / SCROLL_SPEED))
            scroll_prev_y = cy
            clicking = False

        # ── DRAG (fist) ──
        elif not any(fingers[1:]):
            cv2.putText(frame, "DRAG MODE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
            pyautogui.mouseDown()
            wrist = lm[9]
            x = np.interp(wrist.x, [FRAME_REDUCTION/w, 1 - FRAME_REDUCTION/w], [0, screen_w])
            y = np.interp(wrist.y, [FRAME_REDUCTION/h, 1 - FRAME_REDUCTION/h], [0, screen_h])
            pyautogui.moveTo(x, y)

        else:
            pyautogui.mouseUp()
            scroll_prev_y = None

        # Finger indicators at bottom
        labels = ["T", "I", "M", "R", "P"]
        for i, (label, state) in enumerate(zip(labels, fingers)):
            color = (0, 255, 0) if state else (0, 0, 200)
            cv2.putText(frame, label, (10 + i*25, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame, "No hand detected - show hand to camera",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        pyautogui.mouseUp()

    cv2.imshow("Virtual Mouse  |  Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
print("Virtual Mouse stopped.")