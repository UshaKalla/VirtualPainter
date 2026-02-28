import cv2
import numpy as np
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None

palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 0)]
labels = ['Red', 'Green', 'Blue', 'Yellow', 'Eraser']
color_index = 0

brush_thickness = 7
eraser_thickness = 40
prev_x, prev_y = 0, 0

save_counter = 1
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    if canvas is None:
        canvas = np.zeros(h, w, 3, dtype=np.uint8)

    for i, color in enumerate(palette):
        x_start = i * 100
        x_end = (i + 1) * 100
        cv2.rectangle(frame, (x_start, 0), (x_end, 100), color, -1)
        cv2.putText(frame, labels[i], (x_start + 10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 255, 255), 2) 
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
