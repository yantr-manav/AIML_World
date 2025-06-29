# Voice Recognition Troubleshooting and Setup

## Common Issues and Solutions:

### 1. Missing Dependencies
# Install required packages:
# pip install SpeechRecognition pyttsx3 psutil pyaudio

### 2. PyAudio Installation Issues (Common on Windows/Mac)
# Windows:
# pip install pipwin
# pipwin install pyaudio

# Mac:
# brew install portaudio
# pip install pyaudio

# Linux:
# sudo apt-get install python3-pyaudio

### 3. Microphone Permissions
# Make sure Python has microphone access in your system settings

### 4. Test Your Microphone Setup
import speech_recognition as sr
import time

def test_microphone():
    """Test microphone setup"""
    r = sr.Recognizer()
    
    # List available microphones
    print("Available microphones:")
    for i, microphone_name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{i}: {microphone_name}")
    
    # Test default microphone
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source, duration=2)
            print(f"Ambient noise level: {r.energy_threshold}")
            
            print("Say something!")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
        print("Processing...")
        try:
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            return True
        except sr.UnknownValueError:
            print("Could not understand audio")
            return False
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return False
            
    except Exception as e:
        print(f"Microphone error: {e}")
        return False

# Run the test
if __name__ == "__main__":
    test_microphone()

### 5. Enhanced Voice Recognition Class with Better Error Handling
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time

class ImprovedVoiceRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.voice_queue = queue.Queue()
        self.listening = False
        self.voice_thread = None
        
        # TTS setup
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)
        
        # Initialize microphone with error handling
        self.setup_microphone()
    
    def setup_microphone(self):
        """Setup microphone with error handling"""
        try:
            # Try to find the best microphone
            mic_list = sr.Microphone.list_microphone_names()
            print(f"Found {len(mic_list)} microphones")
            
            # Use default microphone
            self.microphone = sr.Microphone()
            
            # Calibrate
            print("Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print(f"Microphone calibrated. Energy threshold: {self.recognizer.energy_threshold}")
            
            return True
            
        except Exception as e:
            print(f"Microphone setup failed: {e}")
            print("Possible solutions:")
            print("1. Check if microphone is connected")
            print("2. Check microphone permissions")
            print("3. Install pyaudio: pip install pyaudio")
            return False
    
    def speak(self, text):
        """Text-to-speech with error handling"""
        try:
            def _speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            threading.Thread(target=_speak, daemon=True).start()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def listen_for_voice_commands(self):
        """Improved voice listening with better error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.listening and consecutive_errors < max_consecutive_errors:
            try:
                with self.microphone as source:
                    # Shorter timeout for better responsiveness
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=4)
                
                try:
                    # Try Google first, fallback to other services if needed
                    command = self.recognizer.recognize_google(audio, language='en-US').lower()
                    print(f"Heard: {command}")
                    self.voice_queue.put(command)
                    consecutive_errors = 0  # Reset error counter on success
                    
                except sr.UnknownValueError:
                    # This is normal - just means no clear speech was detected
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition service error: {e}")
                    consecutive_errors += 1
                    time.sleep(2)  # Wait before retrying
                    
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                pass
            except Exception as e:
                print(f"Unexpected voice recognition error: {e}")
                consecutive_errors += 1
                time.sleep(1)
        
        if consecutive_errors >= max_consecutive_errors:
            print("Too many consecutive errors. Voice recognition stopped.")
            self.listening = False
    
    def start_listening(self):
        """Start voice recognition thread"""
        if self.microphone is None:
            print("Cannot start listening - microphone not initialized")
            return False
        
        self.listening = True
        self.voice_thread = threading.Thread(target=self.listen_for_voice_commands, daemon=True)
        self.voice_thread.start()
        print("Voice recognition started")
        return True
    
    def stop_listening(self):
        """Stop voice recognition"""
        self.listening = False
        if self.voice_thread:
            self.voice_thread.join(timeout=2)
        print("Voice recognition stopped")
    
    def get_command(self):
        """Get the next voice command from queue"""
        try:
            return self.voice_queue.get_nowait()
        except queue.Empty:
            return None
    
    def test_voice_recognition(self):
        """Test voice recognition system"""
        print("Testing voice recognition...")
        print("Say 'hello' to test...")
        
        self.start_listening()
        
        # Wait for command or timeout
        timeout = time.time() + 10  # 10 second timeout
        while time.time() < timeout:
            command = self.get_command()
            if command:
                print(f"Successfully heard: {command}")
                self.speak(f"I heard you say {command}")
                self.stop_listening()
                return True
            time.sleep(0.1)
        
        print("Test timeout - no voice detected")
        self.stop_listening()
        return False

# Test the improved voice recognition
if __name__ == "__main__":
    voice = ImprovedVoiceRecognition()
    voice.test_voice_recognition()