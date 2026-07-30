# Virtual Painter: Air-Drawing with Computer Vision

> **Note:** This project is an interactive computer vision application that lets users draw on a digital canvas in real time using hand gestures captured through a webcam.

---

### Project Summary
* **Inspiration:** Created to explore human-computer interaction (HCI) by combining computer vision and machine learning for a touchless drawing experience.
* **Core Purpose:** To track hand movements via webcam, enabling users to draw, select colors, erase, and save artwork entirely through hand gestures without touching a screen or mouse.

---

### Key Features & Components

* **Real-Time Hand Tracking:** Powered by **MediaPipe Hands** to detect hand landmarks and track movement accurately.
* **Contactless Index Finger Brush:** Uses the position of the index fingertip as a virtual paintbrush.
* **Interactive Color Palette:** A top-screen menu allows users to switch between colors (red, green, blue, yellow) or select an eraser simply by moving their finger over the desired block.
* **Side-by-Side Visualization:** Displays the live webcam feed (with landmark overlays) alongside the separate drawing canvas in a single window.
* **File Management:** Save completed artwork instantly as a PNG image by pressing the `s` key.
* **Quick Exit:** Close the application at any time by pressing the `q` key.

---

### Technologies Used
* **Python** – Core programming language.
* **OpenCV** – Webcam video capture and image processing.
* **MediaPipe** – Real-time hand landmark detection and tracking.
* **NumPy** – Digital canvas creation, manipulation, and layer blending.

---

### Controls & Shortcuts
* **Draw:** Move your index finger across the screen.
* **Select Color / Eraser:** Hover your index finger over the corresponding block in the top color palette.
* **Save Artwork:** Press `s` on your keyboard.
* **Exit Application:** Press `q` on your keyboard.
