import cv2
import mediapipe as mp
import pyautogui


cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0
while True:
    data,image = cap.read()
    image = cv2.flip(image,1)
    image_height, image_width,data = image.shape
    
    
    rgb_frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image,hand,mphands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                pass
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                
                if id == 8:
                    cv2.circle(img=image, center=(x,y),radius=22,color = (0,255,255))
                    index_x = screen_width/image_width*x
                    index_y = screen_height/image_height*y
                    pyautogui.moveTo(index_x,index_y)
                    
                if id == 4:
                    cv2.circle(img=image, center=(x,y),radius=22,color = (0,255,255))
                    thumb_x = screen_width/image_width*x
                    thumb_y = screen_height/image_height*y
                    print('outside: ',abs(index_y-thumb_y))
                    if abs(index_y- thumb_y)<20:
                     pyautogui.click()
                     pyautogui.sleep(1)
                     
                
    image = cv2.resize(image, (800, 600))
    cv2.imshow('Virtual Mouse',image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break