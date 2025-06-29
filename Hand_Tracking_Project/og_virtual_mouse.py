import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
from collections import deque
import threading

class ClearGestureController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Smoothing for cursor movement
        self.cursor_smoothing = 5
        self.cursor_history = deque(maxlen=self.cursor_smoothing)
        
        # State management
        self.dragging = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5
        self.current_gesture = None
        self.gesture_hold_time = 0
        self.gesture_hold_threshold = 0.3  # Hold gesture for 0.3s to activate
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
    def distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def get_landmark_coords(self, landmarks, image_width, image_height):
        return [(int(l.x * image_width), int(l.y * image_height)) for l in landmarks]
    
    def smooth_cursor_movement(self, x, y):
        """Apply smoothing to cursor movement"""
        self.cursor_history.append((x, y))
        if len(self.cursor_history) == self.cursor_smoothing:
            avg_x = sum(pos[0] for pos in self.cursor_history) / self.cursor_smoothing
            avg_y = sum(pos[1] for pos in self.cursor_history) / self.cursor_smoothing
            return int(avg_x), int(avg_y)
        return x, y
    
    def is_finger_up(self, coords, tip_idx, pip_idx):
        """Check if finger is pointing up"""
        return coords[tip_idx][1] < coords[pip_idx][1]
    
    def is_thumb_up(self, coords):
        """Check if thumb is up (pointing right for flipped camera)"""
        return coords[4][0] > coords[3][0]
    
    def get_gesture_name(self, coords):
        """Recognize clear, distinct gestures"""
        # Get finger states
        thumb_up = self.is_thumb_up(coords)
        index_up = self.is_finger_up(coords, 8, 6)
        middle_up = self.is_finger_up(coords, 12, 10)
        ring_up = self.is_finger_up(coords, 16, 14)
        pinky_up = self.is_finger_up(coords, 20, 18)
        
        # Count extended fingers
        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        up_count = sum(fingers_up)
        
        # CURSOR CONTROL: Only index finger up
        if fingers_up == [False, True, False, False, False]:
            return "cursor_control"
        
        # LEFT CLICK: Thumbs up gesture
        elif fingers_up == [True, False, False, False, False]:
            return "left_click"
        
        # RIGHT CLICK: Peace sign (index + middle)
        elif fingers_up == [False, True, True, False, False]:
            return "right_click"
        
        # SCROLL DOWN: Three fingers (index, middle, ring)
        elif fingers_up == [False, True, True, True, False]:
            return "scroll_down"
        
        # SCROLL UP: Four fingers (no thumb)
        elif fingers_up == [False, True, True, True, True]:
            return "scroll_up"
        
        # DRAG START: All five fingers extended (open palm)
        elif fingers_up == [True, True, True, True, True]:
            return "drag_start"
        
        # DRAG END / STOP: Closed fist
        elif up_count == 0:
            return "drag_end"
        
        # NEXT TAB: Pinky up only
        elif fingers_up == [False, False, False, False, True]:
            return "next_tab"
        
        # PREVIOUS TAB: Ring finger up only
        elif fingers_up == [False, False, False, True, False]:
            return "prev_tab"
        
        # WINDOW SWITCH: Middle finger up only
        elif fingers_up == [False, False, True, False, False]:
            return "window_switch"
        
        # SCREENSHOT: Thumb + pinky (call me gesture)
        elif fingers_up == [True, False, False, False, True]:
            return "screenshot"
        
        # DESKTOP LEFT: Thumb + index (L shape)
        elif fingers_up == [True, True, False, False, False]:
            return "desktop_left"
        
        # DESKTOP LEFT: Thumb + index  + middle(L shape)
        elif fingers_up == [True, True, True, False, False]:
            return "desktop_right"
        
        # VOLUME UP: Thumb + middle
        elif fingers_up == [True, False, True, False, False]:
            return "volume_up"
        
        # VOLUME DOWN: Thumb + ring
        elif fingers_up == [True, False, False, True, False]:
            return "volume_down"
        
        else:
            return "unknown"
    
    def execute_gesture_action(self, gesture_name):
        """Execute the action for a recognized gesture"""
        actions = {
            "left_click": lambda: pyautogui.click(),
            "right_click": lambda: pyautogui.rightClick(),
            "scroll_down": lambda: pyautogui.scroll(-500),
            "scroll_up": lambda: pyautogui.scroll(500),
            "next_tab": lambda: pyautogui.hotkey('ctrl', 'tab'),
            "prev_tab": lambda: pyautogui.hotkey('ctrl', 'shift', 'tab'),
            "window_switch": lambda: pyautogui.hotkey('alt', 'tab'),
            "screenshot": lambda: pyautogui.hotkey('win', 'shift', 's'),
            "desktop_left": lambda: pyautogui.hotkey('ctrl', 'win', 'left'),
            "desktop_right": lambda: pyautogui.hotkey('ctrl', 'win', 'right'),
            "volume_up": lambda: pyautogui.press('volumeup'),
            "volume_down": lambda: pyautogui.press('volumedown'),
        }
        
        if gesture_name in actions:
            threading.Thread(target=actions[gesture_name], daemon=True).start()
    
    def process_gestures(self, coords):
        """Process gesture recognition with hold-to-activate system"""
        gesture_name = self.get_gesture_name(coords)
        current_time = time.time()
        
        # Handle cursor control (no hold needed)
        if gesture_name == "cursor_control":
            self.current_gesture = "cursor_control"
            return
        
        # Handle drag system
        if gesture_name == "drag_start" and not self.dragging:
            if self.current_gesture == "drag_start":
                if current_time - self.gesture_hold_time >= self.gesture_hold_threshold:
                    pyautogui.mouseDown()
                    self.dragging = True
                    self.current_gesture = "dragging"
            else:
                self.current_gesture = "drag_start"
                self.gesture_hold_time = current_time
            return
        
        elif gesture_name == "drag_end" and self.dragging:
            pyautogui.mouseUp()
            self.dragging = False
            self.current_gesture = None
            return
        
        # Handle other gestures with hold-to-activate
        if gesture_name != "unknown" and gesture_name != self.current_gesture:
            self.current_gesture = gesture_name
            self.gesture_hold_time = current_time
            
        elif gesture_name == self.current_gesture and not self.dragging:
            # Check if gesture has been held long enough
            if current_time - self.gesture_hold_time >= self.gesture_hold_threshold:
                # Check cooldown
                if current_time - self.last_gesture_time >= self.gesture_cooldown:
                    self.execute_gesture_action(gesture_name)
                    self.last_gesture_time = current_time
                    self.current_gesture = None
        
        elif gesture_name == "unknown":
            self.current_gesture = None
    
    def draw_gesture_guide(self, frame):
        """Draw gesture guide on frame"""
        h, w = frame.shape[:2]
        
        # Background for gesture guide
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-400, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Gesture instructions
        gestures = [
            ("👆 Index Only", "Cursor Control"),
            ("👍 Thumb Up", "Left Click"),
            ("✌️ Peace Sign", "Right Click"),
            ("🤟 3 Fingers", "Scroll Down"),
            ("🖐️ 4 Fingers", "Scroll Up"),
            ("🖐️ Open Palm", "Start Drag"),
            ("✊ Fist", "End Drag"),
            ("🤙 Call Me", "Screenshot"),
            ("🤞 Thumb+Index", "Desktop Left"),
            ("🤞 Thumb down", "Desktop Right"),
            ("🤘 Pinky Only", "Next Tab"),
            ("💍 Ring Only", "Prev Tab"),
            ("🖕 Middle Only", "Alt+Tab"),
            ("👍+Middle", "Volume Up"),
            ("👍+Ring", "Volume Down")
        ]
        
        y_start = 30
        for i, (gesture, action) in enumerate(gestures):
            y = y_start + i * 25
            cv2.putText(frame, f"{gesture}: {action}", (w-390, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_ui(self, frame, coords):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Draw FPS
        cv2.putText(frame, f'FPS: {self.current_fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current gesture with hold progress
        if self.current_gesture:
            current_time = time.time()
            if self.current_gesture in ["cursor_control", "dragging"]:
                color = (0, 255, 0)  # Green for active states
                text = f'Active: {self.current_gesture}'
            else:
                # Show hold progress
                hold_progress = min((current_time - self.gesture_hold_time) / self.gesture_hold_threshold, 1.0)
                color = (0, int(255 * hold_progress), int(255 * (1 - hold_progress)))
                text = f'Gesture: {self.current_gesture} ({int(hold_progress * 100)}%)'
            
            cv2.putText(frame, text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw finger states for debugging
        if coords:
            thumb_up = self.is_thumb_up(coords)
            index_up = self.is_finger_up(coords, 8, 6)
            middle_up = self.is_finger_up(coords, 12, 10)
            ring_up = self.is_finger_up(coords, 16, 14)
            pinky_up = self.is_finger_up(coords, 20, 18)
            
            finger_states = f"T:{int(thumb_up)} I:{int(index_up)} M:{int(middle_up)} R:{int(ring_up)} P:{int(pinky_up)}"
            cv2.putText(frame, finger_states, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw gesture guide
        self.draw_gesture_guide(frame)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
    
    def run(self):
        """Main loop"""
        print("Clear Gesture Controller Started!")
        print("Each gesture must be held for 0.3 seconds to activate")
        print("Press ESC to exit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame and get dimensions
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            result = self.hands.process(rgb_frame)
            coords = None
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get coordinates
                    coords = self.get_landmark_coords(hand_landmarks.landmark, w, h)
                    
                    # Process gestures
                    self.process_gestures(coords)
                    
                    # Move cursor only when in cursor control mode
                    if self.current_gesture == "cursor_control":
                        cursor_x = int(coords[8][0] * self.screen_width / w)
                        cursor_y = int(coords[8][1] * self.screen_height / h)
                        smooth_x, smooth_y = self.smooth_cursor_movement(cursor_x, cursor_y)
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            # Draw UI
            self.draw_ui(frame, coords)
            
            # Update FPS
            self.update_fps()
            
            # Display frame
            cv2.imshow('Clear Gesture Controller - Hold gestures for 0.3s', frame)
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.dragging:
            pyautogui.mouseUp()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Clear Gesture Controller stopped.")

# Run the clear gesture controller
if __name__ == "__main__":
    controller = ClearGestureController()
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.cleanup()