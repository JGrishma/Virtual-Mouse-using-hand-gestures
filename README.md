# 🖐️ Virtual Mouse Using Hand Gestures

Control your computer mouse using hand gestures through your webcam — no physical mouse needed!  
Built with Python, OpenCV, and MediaPipe.

---

## 📋 Requirements

- Python 3.8+
- Webcam

---

## ⚙️ Installation

**1. Download the code**

- Click the green **`<> Code`** button at the top of this page
- Select **`Download ZIP`**
- Extract the ZIP file to a folder (e.g. `C:\Users\YourName\Desktop\VirtualMouse`)

**2. Open the folder in PowerShell**

- Open the extracted folder
- Click the address bar at the top, type `powershell` and press Enter

**3. Install required libraries**

```
py -m pip install opencv-python mediapipe pyautogui numpy pycaw comtypes screen-brightness-control
```

**4. Run the project**

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

- **Python**
- **OpenCV** — webcam feed
- **MediaPipe** — hand tracking
- **PyAutoGUI** — mouse control
- **pycaw** — volume control
- **screen-brightness-control** — brightness control

---

## 📁 Project Structure

```
virtual-mouse/
│
├── virtual_mouse.py        # Main script
├── hand_landmarker.task    # MediaPipe model (auto-downloaded)
└── README.md
```
