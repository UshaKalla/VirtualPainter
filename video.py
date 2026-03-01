import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Canvas where the drawing will happen
canvas = None

# Color palette for drawing
palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 0)]  # Red, Green, Blue, Yellow, Eraser (Black)
labels = ['Red', 'Green', 'Blue', 'Yellow', 'Eraser']
color_index = 0  # Default color index (Red)

# Brush and eraser thickness
brush_thickness = 7
eraser_thickness = 40
prev_x, prev_y = 0, 0  # Store previous coordinates

# Save counter for images
save_counter = 1

while True:
    success, frame = cap.read()  # Capture frame from webcam
    if not success:
        break
    
    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Get the height and width of the frame
    h, w, c = frame.shape
    
    # Initialize the canvas if it hasn't been initialized yet
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw the color palette at the top of the screen
    for i, color in enumerate(palette):
        x_start = i * 100
        x_end = (i + 1) * 100
        cv2.rectangle(frame, (x_start, 0), (x_end, 100), color, -1)  # Draw color blocks
        cv2.putText(frame, labels[i], (x_start + 10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 255, 255), 2)  # Add label for each color block

    # Convert frame to RGB (required by MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)  # Process the frame to detect hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))  # Convert normalized to pixel coordinates

            # Get the index finger coordinates (landmark 8)
            index_x, index_y = lm_list[8]

            # Check if the index finger is in the palette region
            if index_y < 100:  # Palette area is at the top
                color_index = min(index_x // 100, len(palette) - 1)  # Determine color based on X position
                prev_x, prev_y = 0, 0  # Reset previous coordinates

            else:  # Drawing mode
                if prev_x == 0 and prev_y == 0:  # Initialize drawing
                    prev_x, prev_y = index_x, index_y

                # Draw the line on the canvas
                if 0 <= color_index < len(palette):
                    if color_index == len(palette) - 1:  # Eraser mode
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 0), eraser_thickness)
                    else:  # Normal drawing mode
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), palette[color_index], brush_thickness)
                prev_x, prev_y = index_x, index_y

            # Draw the hand landmarks on the frame for visual feedback
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    else:
        prev_x, prev_y = 0, 0  # Reset previous coordinates if no hand is detected
    
    # Combine the frame and canvas (side by side)
    combined = np.hstack((frame, canvas))
    
    # Show the combined output in a window
    cv2.imshow("Virtual Painter", combined)
    
    # Handle key presses for quitting or saving
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('s'):  # Press 's' to save the drawing
        save_path = os.path.join(os.getcwd(), f'painting_{save_counter}.png')
        cv2.imwrite(save_path, canvas)
        print(f"Saved painting as {save_path}")
        save_counter += 1

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()