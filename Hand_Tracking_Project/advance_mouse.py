import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
pyautogui.FAILSAFE = False  # Prevents crash on cursor corner movement

# Initialize hand detector
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Get screen size
screen_width, screen_height = pyautogui.size()

# Variables to store fingertip positions
index_x = index_y = 0
thumb_x = thumb_y = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    image_height, image_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_frame)
    multi_landmarks = results.multi_hand_landmarks

    if multi_landmarks:
        for hand_landmarks in multi_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Loop over each landmark
            landmarks = hand_landmarks.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)

                # Track index finger (id=8)
                if id == 8:
                    index_x = int(screen_width * landmark.x)
                    index_y = int(screen_height * landmark.y)
                    cv2.circle(frame, (x, y), 12, (255, 0, 255), -1)
                    pyautogui.moveTo(index_x, index_y)

                # Track thumb (id=4)
                if id == 4:
                    thumb_x = int(screen_width * landmark.x)
                    thumb_y = int(screen_height * landmark.y)
                    cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)

            # Check if fingers are close → perform click
            distance = abs(index_y - thumb_y)
            if distance < 40:
                pyautogui.click()
                print("Click triggered!")
                time.sleep(1)  # Delay to avoid multiple rapid clicks

    # Resize for better UI display
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow('Advanced Virtual Mouse', frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
