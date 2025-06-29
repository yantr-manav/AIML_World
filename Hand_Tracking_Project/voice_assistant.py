import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
from collections import deque
import threading
import speech_recognition as sr
import pyttsx3
import queue
import psutil
import webbrowser
import os
import subprocess
import json
from datetime import datetime

class IntelligentGestureAgent:
    def __init__(self):
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
            model_complexity=1  # Optimized for performance
        )
        
        # Webcam with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Voice recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_queue = queue.Queue()
        self.voice_thread = None
        self.listening = False
        
        # Text-to-speech setup
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.8)
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Agent state management
        self.agent_active = False
        self.gesture_active = False
        self.voice_commands_active = True
        
        # Enhanced cursor smoothing
        self.cursor_smoothing = 7
        self.cursor_history = deque(maxlen=self.cursor_smoothing)
        self.cursor_acceleration = 1.2
        
        # Gesture state management
        self.dragging = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.4
        self.current_gesture = None
        self.gesture_hold_time = 0
        self.gesture_hold_threshold = 0.25  # Reduced for better responsiveness
        
        # Performance optimization
        self.frame_skip = 0
        self.process_every_n_frames = 2  # Process every 2nd frame for better performance
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Agent capabilities
        self.commands_history = []
        self.user_preferences = self.load_preferences()
        
        # Initialize voice recognition calibration
        self.calibrate_microphone()
        
    def load_preferences(self):
        """Load user preferences from file"""
        try:
            with open('agent_preferences.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            default_prefs = {
                'cursor_sensitivity': 1.0,
                'gesture_hold_time': 0.25,
                'voice_activation_phrase': 'hey agent',
                'auto_screenshots': True,
                'smart_window_management': True
            }
            self.save_preferences(default_prefs)
            return default_prefs
    
    def save_preferences(self, prefs):
        """Save user preferences to file"""
        with open('agent_preferences.json', 'w') as f:
            json.dump(prefs, f, indent=2)
    
    def speak(self, text):
        """Text-to-speech with threading to avoid blocking"""
        def _speak():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
        threading.Thread(target=_speak, daemon=True).start()
    
    def calibrate_microphone(self):
        """Calibrate microphone for better voice recognition"""
        print("Initializing voice recognition...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Voice recognition ready!")
    
    def listen_for_voice_commands(self):
        """Continuous voice command listening"""
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen with timeout to prevent blocking
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    self.voice_queue.put(command)
                except sr.UnknownValueError:
                    pass  # Ignore unrecognized speech
                except sr.RequestError:
                    print("Voice recognition service temporarily unavailable")
                    time.sleep(5)
            except sr.WaitTimeoutError:
                pass  # Continue listening
            except Exception as e:
                print(f"Voice recognition error: {e}")
                time.sleep(1)
    
    def process_voice_commands(self, command):
        """Process voice commands with intelligent responses"""
        command = command.lower().strip()
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Activation commands
        if any(phrase in command for phrase in ['hey agent', 'activate agent', 'start agent']):
            if not self.agent_active:
                self.agent_active = True
                self.gesture_active = True
                self.speak("Agent activated. Gesture control is now active.")
                print(f"[{current_time}] Agent activated")
            return
        
        # Only process other commands if agent is active
        if not self.agent_active:
            return
        
        # Deactivation commands
        if any(phrase in command for phrase in ['deactivate agent', 'stop agent', 'turn off agent', 'goodbye agent']):
            self.agent_active = False
            self.gesture_active = False
            self.speak("Agent deactivated. See you later!")
            print(f"[{current_time}] Agent deactivated")
            return
        
        # Gesture control commands
        if 'enable gestures' in command or 'turn on gestures' in command:
            self.gesture_active = True
            self.speak("Gesture control enabled")
        elif 'disable gestures' in command or 'turn off gestures' in command:
            self.gesture_active = False
            self.speak("Gesture control disabled")
        
        # System commands
        elif 'open browser' in command or 'launch chrome' in command:
            webbrowser.open('https://www.google.com')
            self.speak("Opening browser")
        
        elif 'take screenshot' in command:
            pyautogui.screenshot(f'screenshot_{int(time.time())}.png')
            self.speak("Screenshot taken")
        
        elif 'system status' in command or 'how is system' in command:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            self.speak(f"CPU usage is {cpu_percent}%. Memory usage is {memory_percent}%")
        
        elif 'close window' in command:
            pyautogui.hotkey('alt', 'f4')
            self.speak("Closing window")
        
        elif 'minimize all' in command:
            pyautogui.hotkey('win', 'd')
            self.speak("Minimizing all windows")
        
        elif 'new tab' in command:
            pyautogui.hotkey('ctrl', 't')
            self.speak("Opening new tab")
        
        elif 'close tab' in command:
            pyautogui.hotkey('ctrl', 'w')
            self.speak("Closing tab")
        
        elif 'volume up' in command:
            for _ in range(5):
                pyautogui.press('volumeup')
            self.speak("Volume increased")
        
        elif 'volume down' in command:
            for _ in range(5):
                pyautogui.press('volumedown')
            self.speak("Volume decreased")
        
        elif 'mute' in command:
            pyautogui.press('volumemute')
            self.speak("Audio muted")
        
        elif 'what time' in command:
            current_time_speak = datetime.now().strftime("%I:%M %p")
            self.speak(f"The time is {current_time_speak}")
        
        elif 'help' in command or 'what can you do' in command:
            self.speak("I can control gestures, take screenshots, manage windows, control volume, and much more. Just ask me!")
        
        # Add command to history
        self.commands_history.append({
            'command': command,
            'timestamp': current_time,
            'executed': True
        })
    
    def distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def get_landmark_coords(self, landmarks, image_width, image_height):
        """Get landmark coordinates"""
        return [(int(l.x * image_width), int(l.y * image_height)) for l in landmarks]
    
    def smooth_cursor_movement(self, x, y):
        """Enhanced cursor smoothing with acceleration"""
        self.cursor_history.append((x, y))
        if len(self.cursor_history) == self.cursor_smoothing:
            # Calculate smoothed position
            avg_x = sum(pos[0] for pos in self.cursor_history) / self.cursor_smoothing
            avg_y = sum(pos[1] for pos in self.cursor_history) / self.cursor_smoothing
            
            # Apply sensitivity from preferences
            sensitivity = self.user_preferences.get('cursor_sensitivity', 1.0)
            return int(avg_x * sensitivity), int(avg_y * sensitivity)
        return x, y
    
    def is_finger_up(self, coords, tip_idx, pip_idx):
        """Check if finger is pointing up"""
        return coords[tip_idx][1] < coords[pip_idx][1]
    
    def is_thumb_up(self, coords):
        """Check if thumb is up"""
        return coords[4][0] > coords[3][0]
    
    def get_gesture_name(self, coords):
        """Enhanced gesture recognition with more gestures"""
        # Get finger states
        thumb_up = self.is_thumb_up(coords)
        index_up = self.is_finger_up(coords, 8, 6)
        middle_up = self.is_finger_up(coords, 12, 10)
        ring_up = self.is_finger_up(coords, 16, 14)
        pinky_up = self.is_finger_up(coords, 20, 18)
        
        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        up_count = sum(fingers_up)
        
        # Enhanced gesture set
        gesture_map = {
            (False, True, False, False, False): "cursor_control",
            (True, False, False, False, False): "left_click",
            (False, True, True, False, False): "right_click",
            (False, True, True, True, False): "scroll_down",
            (False, True, True, True, True): "scroll_up",
            (True, True, True, True, True): "drag_start",
            (False, False, False, False, False): "drag_end",
            (False, False, False, False, True): "next_tab",
            (False, False, False, True, False): "prev_tab",
            (False, False, True, False, False): "window_switch",
            (True, False, False, False, True): "screenshot",
            (True, True, False, False, False): "desktop_left",
            (True, True, True, False, False): "desktop_right",
            (True, False, True, False, False): "volume_up",
            (True, False, False, True, False): "volume_down",
            (False, True, False, True, False): "copy",  # Index + Ring
            (False, False, True, True, False): "paste",  # Middle + Ring
            (True, True, False, True, False): "undo",  # Thumb + Index + Ring
        }
        
        return gesture_map.get(tuple(fingers_up), "unknown")
    
    def execute_gesture_action(self, gesture_name):
        """Execute gesture actions with enhanced functionality"""
        actions = {
            "left_click": lambda: pyautogui.click(),
            "right_click": lambda: pyautogui.rightClick(),
            "scroll_down": lambda: pyautogui.scroll(-500),
            "scroll_up": lambda: pyautogui.scroll(500),
            "next_tab": lambda: pyautogui.hotkey('ctrl', 'tab'),
            "prev_tab": lambda: pyautogui.hotkey('ctrl', 'shift', 'tab'),
            "window_switch": lambda: pyautogui.hotkey('alt', 'tab'),
            "screenshot": lambda: self.take_smart_screenshot(),
            "desktop_left": lambda: pyautogui.hotkey('ctrl', 'win', 'left'),
            "desktop_right": lambda: pyautogui.hotkey('ctrl', 'win', 'right'),
            "volume_up": lambda: [pyautogui.press('volumeup') for _ in range(3)],
            "volume_down": lambda: [pyautogui.press('volumedown') for _ in range(3)],
            "copy": lambda: pyautogui.hotkey('ctrl', 'c'),
            "paste": lambda: pyautogui.hotkey('ctrl', 'v'),
            "undo": lambda: pyautogui.hotkey('ctrl', 'z'),
        }
        
        if gesture_name in actions:
            threading.Thread(target=actions[gesture_name], daemon=True).start()
    
    def take_smart_screenshot(self):
        """Take screenshot with timestamp and optional voice confirmation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pyautogui.screenshot(filename)
        if self.user_preferences.get('auto_screenshots', True):
            self.speak("Screenshot saved")
    
    def process_gestures(self, coords):
        """Enhanced gesture processing with better state management"""
        if not self.gesture_active:
            return
            
        gesture_name = self.get_gesture_name(coords)
        current_time = time.time()
        
        # Handle cursor control (no hold needed)
        if gesture_name == "cursor_control":
            self.current_gesture = "cursor_control"
            return
        
        # Handle drag system
        if gesture_name == "drag_start" and not self.dragging:
            if self.current_gesture == "drag_start":
                hold_time = self.user_preferences.get('gesture_hold_time', 0.25)
                if current_time - self.gesture_hold_time >= hold_time:
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
            hold_time = self.user_preferences.get('gesture_hold_time', 0.25)
            if current_time - self.gesture_hold_time >= hold_time:
                if current_time - self.last_gesture_time >= self.gesture_cooldown:
                    self.execute_gesture_action(gesture_name)
                    self.last_gesture_time = current_time
                    self.current_gesture = None
        
        elif gesture_name == "unknown":
            self.current_gesture = None
    
    def draw_enhanced_ui(self, frame, coords):
        """Enhanced UI with more information"""
        h, w = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 120), (50, 50, 50), 2)
        
        # Agent status
        status_color = (0, 255, 0) if self.agent_active else (0, 0, 255)
        status_text = "ACTIVE" if self.agent_active else "INACTIVE"
        cv2.putText(frame, f'Agent: {status_text}', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Gesture status
        gesture_color = (0, 255, 0) if self.gesture_active else (0, 255, 255)
        gesture_text = "ON" if self.gesture_active else "OFF"
        cv2.putText(frame, f'Gestures: {gesture_text}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
        
        # FPS and performance
        cv2.putText(frame, f'FPS: {self.current_fps}', (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Current gesture with progress
        if self.current_gesture and self.gesture_active:
            current_time = time.time()
            if self.current_gesture in ["cursor_control", "dragging"]:
                color = (0, 255, 0)
                text = f'▶ {self.current_gesture.replace("_", " ").title()}'
            else:
                hold_time = self.user_preferences.get('gesture_hold_time', 0.25)
                hold_progress = min((current_time - self.gesture_hold_time) / hold_time, 1.0)
                color = (0, int(255 * hold_progress), int(255 * (1 - hold_progress)))
                text = f'⏳ {self.current_gesture.replace("_", " ").title()} ({int(hold_progress * 100)}%)'
            
            cv2.putText(frame, text, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Voice command indicator
        if self.listening:
            cv2.circle(frame, (w-30, 30), 10, (0, 255, 0), -1)
            cv2.putText(frame, 'LISTENING', (w-120, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Gesture guide (compact version)
        self.draw_compact_gesture_guide(frame)
    
    def draw_compact_gesture_guide(self, frame):
        """Draw a compact gesture guide"""
        h, w = frame.shape[:2]
        
        # Compact gesture list
        gestures = [
            "👆: Cursor | 👍: Click | ✌️: Right Click",
            "🤟: Scroll ↓ | 🖐️: Scroll ↑ | ✊: Drag",
            "🤙: Screenshot | 🤞: Desktop | 🤘: Tabs"
        ]
        
        # Background
        guide_height = len(gestures) * 25 + 20
        cv2.rectangle(frame, (w-300, h-guide_height), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (w-300, h-guide_height), (w, h), (100, 100, 100), 1)
        
        # Draw gestures
        for i, gesture_line in enumerate(gestures):
            y = h - guide_height + 25 + i * 25
            cv2.putText(frame, gesture_line, (w-290, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
    
    def run(self):
        """Main agent loop with enhanced functionality"""
        print("🤖 Intelligent Gesture Agent Starting...")
        print("=" * 50)
        print("Voice Commands:")
        print("• 'Hey Agent' - Activate agent and gestures")
        print("• 'Stop Agent' - Deactivate agent")
        print("• 'Enable/Disable Gestures' - Toggle gesture control")
        print("• 'Take Screenshot' - Capture screen")
        print("• 'System Status' - Check CPU/Memory")
        print("• 'What time' - Get current time")
        print("• 'Help' - List capabilities")
        print("=" * 50)
        print("Press ESC to exit")
        
        # Start voice recognition
        self.listening = True
        self.voice_thread = threading.Thread(target=self.listen_for_voice_commands, daemon=True)
        self.voice_thread.start()
        
        # Initial greeting
        self.speak("Intelligent Gesture Agent initialized. Say 'Hey Agent' to activate.")
        
        while True:
            # Process voice commands
            while not self.voice_queue.empty():
                command = self.voice_queue.get()
                self.process_voice_commands(command)
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame and optimize processing
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Skip frames for performance optimization
            self.frame_skip += 1
            process_frame = self.frame_skip % self.process_every_n_frames == 0
            
            coords = None
            if process_frame and self.agent_active:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb_frame)
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Draw hand landmarks only when active
                        if self.gesture_active:
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Get coordinates and process gestures
                        coords = self.get_landmark_coords(hand_landmarks.landmark, w, h)
                        self.process_gestures(coords)
                        
                        # Enhanced cursor control
                        if self.current_gesture == "cursor_control":
                            cursor_x = int(coords[8][0] * self.screen_width / w)
                            cursor_y = int(coords[8][1] * self.screen_height / h)
                            smooth_x, smooth_y = self.smooth_cursor_movement(cursor_x, cursor_y)
                            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            # Draw enhanced UI
            self.draw_enhanced_ui(frame, coords)
            
            # Update performance metrics
            self.update_fps()
            
            # Display frame
            cv2.imshow('🤖 Intelligent Gesture Agent - Voice Activated', frame)
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup"""
        print("Shutting down Intelligent Gesture Agent...")
        self.listening = False
        self.agent_active = False
        self.gesture_active = False
        
        if self.dragging:
            pyautogui.mouseUp()
        
        # Save preferences and command history
        self.save_preferences(self.user_preferences)
        
        # Cleanup resources
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final goodbye
        self.speak("Goodbye! Agent shutdown complete.")
        print("🤖 Intelligent Gesture Agent stopped.")

# Main execution
if __name__ == "__main__":
    print("Installing required packages if not present...")
    try:
        import speech_recognition
        import pyttsx3
        import psutil
    except ImportError:
        print("Please install required packages:")
        print("pip install SpeechRecognition pyttsx3 psutil pyaudio")
        exit(1)
    
    agent = IntelligentGestureAgent()
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        agent.cleanup()