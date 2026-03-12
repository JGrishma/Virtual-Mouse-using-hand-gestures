"""
Virtual Mouse Using Hand Gestures
Compatible with mediapipe 0.10.32 (Tasks API)
Features: Move, Click, Right Click, Scroll, Drag,
          Volume Control, Brightness Control, Screenshot
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

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    VOLUME_AVAILABLE = True
except:
    VOLUME_AVAILABLE = False

try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except:
    BRIGHTNESS_AVAILABLE = False

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
CLICK_THRESHOLD   = 0.05
SCROLL_SPEED      = 20
DOUBLE_CLICK_TIME = 0.3

# ─────────────────────────────────────────────
# VOLUME SETUP
# ─────────────────────────────────────────────
if VOLUME_AVAILABLE:
    try:
        devices    = AudioUtilities.GetSpeakers()
        interface  = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_ctrl= cast(interface, POINTER(IAudioEndpointVolume))
        vol_range  = volume_ctrl.GetVolumeRange()
        VOL_MIN, VOL_MAX = vol_range[0], vol_range[1]
    except:
        VOLUME_AVAILABLE = False

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_x, prev_y      = 0, 0
curr_x, curr_y      = 0, 0
last_click_time     = 0
clicking            = False
scroll_prev_y       = None
latest_result       = None
screenshot_cooldown = 0

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

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def draw_hand(frame, lm_px, color=(0,200,200)):
    for a, b in CONNECTIONS:
        cv2.line(frame, lm_px[a], lm_px[b], color, 2)
    for pt in lm_px:
        cv2.circle(frame, pt, 4, (255,255,255), cv2.FILLED)
        cv2.circle(frame, pt, 4, color, 1)

def fingers_up(lm, handedness="Right"):
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    up = []
    if handedness == "Right":
        up.append(lm[tips[0]].x < lm[joints[0]].x)
    else:
        up.append(lm[tips[0]].x > lm[joints[0]].x)
    for i in range(1, 5):
        up.append(lm[tips[i]].y < lm[joints[i]].y)
    return up

def norm_dist(p1, p2):
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

def draw_bar(frame, x, y, w, h, pct, color, label):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), cv2.FILLED)
    filled = int(h * pct / 100)
    cv2.rectangle(frame, (x, y+h-filled), (x+w, y+h), color, cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{label}: {int(pct)}%",
                (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ─────────────────────────────────────────────
# GESTURE DETECTION — priority ordered
# ─────────────────────────────────────────────
def detect_gesture(fingers, lm):
    """
    Returns gesture name as string.
    Checked in priority order — most specific first.
    """
    thumb, index, middle, ring, pinky = fingers

    # Distances used for multi-finger gestures
    thumb_pinky_dist = norm_dist(lm[4], lm[20])
    thumb_index_dist = norm_dist(lm[4], lm[8])
    index_middle_dist= norm_dist(lm[8], lm[12])

    all_up   = all(fingers)

    # 1. SCREENSHOT — 🤘 Rock sign: index + pinky up, middle + ring + thumb down
    if index and pinky and not middle and not ring and not thumb:
        return "screenshot"

    # 2. BRIGHTNESS — all 5 up
    if all_up:
        return "brightness"

    # 3. VOLUME — only thumb + pinky up
    if thumb and pinky and not index and not middle and not ring:
        return "volume"

    # 4. DRAG — fist (no fingers up)
    if not index and not middle and not ring and not pinky:
        return "drag"

    # 5. RIGHT CLICK — thumb + index only
    if thumb and index and not middle and not ring and not pinky:
        if thumb_index_dist < CLICK_THRESHOLD:
            return "right_click"
        return "right_click_mode"

    # 6. SCROLL — index + middle + ring (3 fingers)
    if index and middle and ring and not pinky:
        return "scroll"

    # 7. LEFT CLICK — index + middle only
    if index and middle and not ring:
        return "click"

    # 8. MOVE — only index up
    if index and not middle:
        return "move"

    return "idle"

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("="*45)
print("  Virtual Mouse — All Gestures")
print("="*45)
print("  Move        : ☝️  Index only")
print("  Click       : ✌️  Index+Middle close")
print("  Right Click : 👌 Thumb+Index pinch")
print("  Scroll      : 🤟 Index+Middle+Ring")
print("  Volume      : 🤙 Thumb+Pinky only")
print("  Brightness  : ✋ All 5 (not spread)")
print("  Screenshot  : 🖐️  All 5 SPREAD wide")
print("  Drag        : ✊ Fist")
print("="*45)
print("Press Q to quit\n")

timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error. Try changing CAMERA_INDEX to 1.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    cv2.rectangle(frame,
                  (FRAME_REDUCTION, FRAME_REDUCTION),
                  (w-FRAME_REDUCTION, h-FRAME_REDUCTION),
                  (0,200,255), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp += 1
    detector.detect_async(mp_image, timestamp)

    if latest_result and latest_result.hand_landmarks:
        lm         = latest_result.hand_landmarks[0]
        handedness = latest_result.handedness[0][0].category_name
        lm_px      = [(int(p.x * w), int(p.y * h)) for p in lm]

        fingers  = fingers_up(lm, handedness)
        gesture  = detect_gesture(fingers, lm)

        index_px  = lm_px[8]
        middle_px = lm_px[12]
        thumb_px  = lm_px[4]
        pinky_px  = lm_px[20]

        # ══════════════════════════════════════
        # 📸 SCREENSHOT
        # ══════════════════════════════════════
        if gesture == "screenshot":
            draw_hand(frame, lm_px, (0,255,100))
            cv2.putText(frame, "SCREENSHOT  🤘", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,100), 2)
            now = time.time()
            if now - screenshot_cooldown > 2:
                screenshot_cooldown = now
                # Save with timestamp so files don't overwrite
                fname = f"screenshot_{int(now)}.png"
                pyautogui.screenshot(fname)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,255,100), -1)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                cv2.putText(frame, f"SAVED: {fname}", (30, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,100), 2)
                cv2.imshow("Virtual Mouse  |  Press Q to quit", frame)
                cv2.waitKey(800)

        # ══════════════════════════════════════
        # 🔆 BRIGHTNESS
        # ══════════════════════════════════════
        elif gesture == "brightness":
            d = norm_dist(lm[4], lm[20])
            bright_pct = float(np.clip(np.interp(d, [0.05, 0.4], [0, 100]), 5, 100))
            if BRIGHTNESS_AVAILABLE:
                try:
                    sbc.set_brightness(int(bright_pct))
                except:
                    pass
            draw_hand(frame, lm_px, (255,220,0))
            draw_bar(frame, w-60, 80, 30, 200, bright_pct, (255,220,0), "BRI")
            cv2.line(frame, thumb_px, pinky_px, (255,220,0), 2)
            cv2.putText(frame, "BRIGHTNESS CONTROL", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,220,0), 2)

        # ══════════════════════════════════════
        # 🔊 VOLUME
        # ══════════════════════════════════════
        elif gesture == "volume":
            d = norm_dist(lm[4], lm[20])
            vol_pct = float(np.clip(np.interp(d, [0.1, 0.6], [0, 100]), 0, 100))
            if VOLUME_AVAILABLE:
                vol_db = np.interp(vol_pct, [0,100], [VOL_MIN, VOL_MAX])
                volume_ctrl.SetMasterVolumeLevel(vol_db, None)
            draw_hand(frame, lm_px, (0,200,255))
            draw_bar(frame, w-60, 80, 30, 200, vol_pct, (0,200,255), "VOL")
            cv2.line(frame, thumb_px, pinky_px, (0,200,255), 2)
            cv2.putText(frame, "VOLUME CONTROL", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

        # ══════════════════════════════════════
        # ☝️ MOVE
        # ══════════════════════════════════════
        elif gesture == "move":
            x = np.interp(lm[8].x, [FRAME_REDUCTION/w, 1-FRAME_REDUCTION/w], [0, screen_w])
            y = np.interp(lm[8].y, [FRAME_REDUCTION/h, 1-FRAME_REDUCTION/h], [0, screen_h])
            curr_x = prev_x + (x - prev_x) / SMOOTHENING
            curr_y = prev_y + (y - prev_y) / SMOOTHENING
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            clicking = False
            scroll_prev_y = None
            draw_hand(frame, lm_px, (0,255,0))
            cv2.circle(frame, index_px, 12, (0,255,0), cv2.FILLED)
            cv2.putText(frame, "MOVE", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # ══════════════════════════════════════
        # ✌️ LEFT CLICK
        # ══════════════════════════════════════
        elif gesture == "click":
            d = norm_dist(lm[8], lm[12])
            draw_hand(frame, lm_px, (255,165,0))
            cv2.putText(frame, "CLICK MODE", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0), 2)
            cv2.line(frame, index_px, middle_px, (255,165,0), 2)
            if d < CLICK_THRESHOLD:
                now = time.time()
                if not clicking:
                    if now - last_click_time < DOUBLE_CLICK_TIME:
                        pyautogui.doubleClick()
                        cv2.putText(frame, "DOUBLE CLICK!", (10,90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    else:
                        pyautogui.click()
                        cv2.putText(frame, "CLICK!", (10,90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    clicking = True
                    last_click_time = now
                cv2.circle(frame, index_px,  12, (0,255,255), cv2.FILLED)
                cv2.circle(frame, middle_px, 12, (0,255,255), cv2.FILLED)
            else:
                clicking = False
            scroll_prev_y = None

        # ══════════════════════════════════════
        # 👌 RIGHT CLICK
        # ══════════════════════════════════════
        elif gesture in ("right_click", "right_click_mode"):
            draw_hand(frame, lm_px, (200,0,255))
            cv2.putText(frame, "RIGHT CLICK MODE", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,255), 2)
            if gesture == "right_click":
                pyautogui.rightClick()
                cv2.putText(frame, "RIGHT CLICK!", (10,90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,255), 2)
                time.sleep(0.4)

        # ══════════════════════════════════════
        # 🤟 SCROLL
        # ══════════════════════════════════════
        elif gesture == "scroll":
            cy = lm[8].y
            draw_hand(frame, lm_px, (255,0,255))
            cv2.putText(frame, "SCROLL MODE", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            if scroll_prev_y is not None:
                delta = scroll_prev_y - cy
                if abs(delta) > 0.01:
                    pyautogui.scroll(int(delta * 500 / SCROLL_SPEED))
            scroll_prev_y = cy
            clicking = False

        # ══════════════════════════════════════
        # ✊ DRAG
        # ══════════════════════════════════════
        elif gesture == "drag":
            draw_hand(frame, lm_px, (0,100,255))
            cv2.putText(frame, "DRAG MODE", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)
            pyautogui.mouseDown()
            x = np.interp(lm[9].x, [FRAME_REDUCTION/w, 1-FRAME_REDUCTION/w], [0, screen_w])
            y = np.interp(lm[9].y, [FRAME_REDUCTION/h, 1-FRAME_REDUCTION/h], [0, screen_h])
            pyautogui.moveTo(x, y)

        else:
            draw_hand(frame, lm_px)
            pyautogui.mouseUp()
            scroll_prev_y = None

        # Finger indicators bottom-left
        labels = ["T","I","M","R","P"]
        for i, (label, state) in enumerate(zip(labels, fingers)):
            color = (0,255,0) if state else (0,0,200)
            cv2.putText(frame, label, (10+i*25, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show current gesture name bottom-right
        cv2.putText(frame, gesture.upper().replace("_"," "),
                    (w-200, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    else:
        cv2.putText(frame, "No hand detected - show hand to camera",
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        pyautogui.mouseUp()

    cv2.imshow("Virtual Mouse  |  Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
print("Virtual Mouse stopped.")
