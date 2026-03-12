# Virtual-Mouse-using-hand-gestures
# 🖐️ Virtual Mouse Using Hand Gestures

Control your computer mouse using hand gestures through your webcam — no physical mouse needed!

Built with Python, OpenCV, and MediaPipe.

---

## 📋 Requirements

- Python 3.8+
- Webcam

---

## ⚙️ Installation

1. Clone the repository
```
git clone https://github.com/JGrishma/Virtual-Mouse-using-hand-gestures
cd virtual-mouse
```

2. Install required libraries
```
py -m pip install opencv-python mediapipe pyautogui numpy pycaw comtypes screen-brightness-control
```

3. Run the project
```
py virtual_mouse.py
```

> The first time you run it, it will automatically download the hand landmark model (~6MB).

---

## 🖐️ Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Move cursor |
| ✌️ Index + Middle fingers close together | Left Click |
| ✌️ Pinch twice quickly | Double Click |
| 👌 Thumb + Index pinch | Right Click |
| 🤟 Index + Middle + Ring fingers up | Scroll |
| ✊ Fist | Drag |
| 🤙 Thumb + Pinky only | Volume Control |
| ✋ All 5 fingers up | Brightness Control |
| 🤘 Index + Pinky up (Rock sign) | Screenshot |

---

## 🛠️ Technologies Used

- [Python](https://python.org)
- [OpenCV](https://opencv.org) — webcam feed
- [MediaPipe](https://mediapipe.dev) — hand tracking
- [PyAutoGUI](https://pyautogui.readthedocs.io) — mouse control
- [pycaw](https://github.com/AndreMiras/pycaw) — volume control
- [screen-brightness-control](https://github.com/Crozzers/screen-brightness-control) — brightness control

---

## 📁 Project Structure

```
virtual-mouse/
│
├── virtual_mouse.py        # Main script
├── hand_landmarker.task    # MediaPipe model (auto-downloaded)
└── README.md
```
