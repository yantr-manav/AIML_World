import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pyautogui.FAILSAFE = False  # Avoid crashes on corner

# Set up webcam with reduced resolution
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for lower latency (Windows)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Hand detection model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Track previous mouse position and click timing
prev_x, prev_y = 0, 0
click_delay = 0.3  # seconds
last_click_time = 0

# Main loop
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_height, image_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # OPTIONAL: draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]
            thumb = landmarks[4]

            # Convert to screen coordinates
            index_x = int(index_finger.x * screen_width)
            index_y = int(index_finger.y * screen_height)
            screen_thumb_y = int(thumb.y * screen_height)

            # Move cursor only if significant change
            if abs(index_x - prev_x) > 5 or abs(index_y - prev_y) > 5:
                pyautogui.moveTo(index_x, index_y)
                prev_x, prev_y = index_x, index_y

            # Draw circles for feedback
            cx = int(index_finger.x * image_width)
            cy = int(index_finger.y * image_height)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)

            tx = int(thumb.x * image_width)
            ty = int(thumb.y * image_height)
            cv2.circle(frame, (tx, ty), 10, (0, 255, 255), -1)

            # Click detection
            if abs(index_y - screen_thumb_y) < 40:
                if time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    print("Click!")
                    last_click_time = time.time()

    # Show FPS
    fps = int(1 / (time.time() - start_time))
    cv2.putText(frame, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Optimized Virtual Mouse', frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
