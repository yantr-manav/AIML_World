import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Helper Functions
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_coords(landmarks, width, height):
    return [(int(l.x * width), int(l.y * height)) for l in landmarks]

# Gesture State
dragging = False
last_gesture_time = 0
cooldown = 0.4  # seconds

def cooldown_elapsed():
    return time.time() - last_gesture_time > cooldown

def update_gesture_time():
    global last_gesture_time
    last_gesture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coords = get_coords(hand_landmarks.landmark, w, h)

            # Finger Tips
            thumb, index, middle, ring, pinky = coords[4], coords[8], coords[12], coords[16], coords[20]

            # Distances
            d_thumb_index = distance(thumb, index)
            d_index_middle = distance(index, middle)
            d_thumb_middle = distance(thumb, middle)
            d_thumb_ring = distance(thumb, ring)
            d_index_ring = distance(index, ring)
            d_thumb_pinky = distance(thumb, pinky)

            # Mouse movement
            pyautogui.moveTo(index[0] * screen_width / w,
                             index[1] * screen_height / h,
                             duration=0.01)

            # Clicks and Actions
            if d_thumb_index < 40 and cooldown_elapsed():
                pyautogui.click()
                update_gesture_time()

            elif d_index_middle < 40 and cooldown_elapsed():
                pyautogui.rightClick()
                update_gesture_time()

            elif d_thumb_middle < 40 and cooldown_elapsed():
                pyautogui.scroll(-50)
                update_gesture_time()

            elif d_thumb_ring < 40 and cooldown_elapsed():
                pyautogui.scroll(50)
                update_gesture_time()

            elif d_index_ring < 40 and cooldown_elapsed():
                pyautogui.hotkey('ctrl', 'tab')
                update_gesture_time()

            elif d_thumb_pinky < 40 and cooldown_elapsed():
                pyautogui.hotkey('alt', 'tab')
                update_gesture_time()

            elif all(distance(coords[i], coords[0]) > 60 for i in [8, 12, 16, 20]) and cooldown_elapsed():
                pyautogui.hotkey('ctrl', 'win', 'right')
                update_gesture_time()

            elif all(distance(coords[i], coords[0]) < 30 for i in [8, 12, 16, 20]) and cooldown_elapsed():
                pyautogui.hotkey('ctrl', 'win', 'left')
                update_gesture_time()

            elif d_thumb_index > 150:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    # Display Feed
    frame = cv2.resize(frame, (1000, 700))
    cv2.imshow('Gesture Controller', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
