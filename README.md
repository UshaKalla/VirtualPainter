This application is a Virtual Painter that allows users to draw on a digital canvas using hand gestures detected through a webcam. It uses OpenCV for video processing, MediaPipe Hands for real-time hand tracking, and NumPy for managing the drawing canvas.

Description

The Virtual Painter captures live video from the webcam and tracks the user's hand movements using MediaPipe's hand landmark detection. The position of the index fingertip acts as a virtual brush, enabling users to draw naturally in the air without touching the screen.

A color palette is displayed at the top of the video feed, allowing users to select different drawing colors (red, green, blue, yellow) or switch to an eraser by moving their index finger over the desired color block. Once a color is selected, moving the index finger across the screen creates smooth lines on a separate drawing canvas.

The application overlays the detected hand landmarks on the live video for visual feedback and displays the webcam feed alongside the drawing canvas in a single window. Users can:

Draw freely using hand movements.
Switch between multiple colors.
Erase parts of the drawing using the eraser tool.
Save the completed artwork as a PNG image by pressing the 's' key.
Exit the application by pressing the 'q' key.
Key Features
Real-time hand tracking using MediaPipe Hands.
Contactless drawing with the index finger.
Interactive color palette for quick color selection.
Eraser mode for correcting drawings.
Live visualization of hand landmarks.
Save drawings as image files.
Side-by-side display of webcam feed and drawing canvas.
Technologies Used
Python
OpenCV – Webcam capture and image processing.
MediaPipe – Hand detection and landmark tracking.
NumPy – Canvas creation and manipulation.
