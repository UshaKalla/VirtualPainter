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
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(palette):
        x_start = i * 100
        x_end = (i + 1) * 100
        cv2.rectangle(frame, (x_start, 0), (x_end, 100), color, -1)
        cv2.putText(frame, labels[i], (x_start + 10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 255, 255), 2) 
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            index_x, index_y = lm_list[8]

            if index_y < 100:
                color_index = min(index_x // 100, len(palette) - 1)
                prev_x, prev_y = 0, 0

            else:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y

                if 0 <= color_index < len(palette):
                    if color_index == len(palette) - 1:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 0), eraser_thickness)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), palette[color_index], brush_thickness)
                prev_x, prev_y = index_x, index_y
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    else:
        prev_x, prev_y = 0, 0
    combined = np.hstack((frame, canvas))
    cv2.imshow("Virtual Painter", combined)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_path = os.path.join(os.getcwd(), f'painting_{save_counter}.png')
        cv2.imwrite(save_path, canvas)
        print(f"Saved painting as {save_path}")
        save_counter += 1
cap.release()
cv2.destroyAllWindows()