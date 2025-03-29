# Hand Tracking with MediaPipe and Robotic Arm Control

## Overview
This project uses MediaPipe to track hand movements, extract 3D coordinates, and translate them into robotic arm movements. The data is smoothed and plotted using Matplotlib for real-time visualization.

## Features
- **Real-time hand tracking** using OpenCV and MediaPipe
- **Smoothing of hand coordinates** to reduce noise
- **Mapping hand movements to robotic arm control**
- **Live plotting of X, Y, and Z coordinates**
- **Fist detection** to determine whether the hand is open or closed

## Installation

### Prerequisites
Ensure you have Python installed (preferably 3.8+).

### Install Dependencies
Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```
Then, install dependencies using the provided `requirements.txt`:
```sh
pip install -r requirements.txt
```

## Usage
Run the script to start real-time hand tracking and robotic arm control:
```sh
python hand_tracking.py
```

### Controls
- **Move your hand** to track its X, Y, Z positions.
- **Close your fist** to stop movement.
- **Open your fist** to allow movement.
- **Press 'q'** to exit the program.

## Screenshots

### Hand Tracking
![Hand Tracking](screenshots/hand_tracking.png)

### Real-time Plot
![Real-time Plot](screenshots/real_time_plot.png)

## File Structure
```
📁 project_root
│-- hand_tracking.py  # Main script for hand tracking and robotic control
│-- requirements.txt  # Dependencies
│-- README.md         # Documentation
│-- screenshots/      # Folder for images and plots
```

## Future Improvements
- Improve hand tracking accuracy
- Optimize robotic arm control for smoother movements
- Support multiple hand gestures for different commands

## License
This project is open-source under the MIT License.

---
Enjoy tracking and automating your robotic arm! 🚀

