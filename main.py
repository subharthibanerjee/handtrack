import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
import json
import time
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

alpha = 0.2  # Smoothing factor (0 < alpha < 1)
smoothed_x, smoothed_y, smoothed_z = None, None, None

# Open Webcam

cap = cv2.VideoCapture(0)
my_url = "http://192.168.4.1/"
def smooth(value, previous):
    """Applies exponential smoothing to reduce noise."""
    if previous is None:
        return value
    return alpha * value + (1 - alpha) * previous

def create_json_commands(url, cmd):
    #print(f"Sending command {cmd}")
    cmd = json.dumps(cmd)
    print(cmd)
    url = f"{url}js?json={cmd}"
    resp = requests.get(url)
    print(f"got response {resp.status_code}")
    data = resp.text
def control_single_joint(url, joint=0, rad=0, spd=0, acc=10):
    CMD_JOINT_CTRL = {"T":101,"joint":joint,"rad":rad,"spd":spd,"acc":acc}
    create_json_commands(url, CMD_JOINT_CTRL)
def joint_ctrl(url, base=0, shoulder=0, elbow=1.57,hand=0, spd=0,acc=10):
    CMD_JOINTS_RAD_CTRL = {"T":102,"base":base,"shoulder":shoulder,"elbow":elbow,"hand":hand,"spd":spd,"acc":10}
    create_json_commands(url, CMD_JOINTS_RAD_CTRL)
def move_to_xyzt(url, t, x, y, z, tor=3.14, spd=10):
    CMD_XYZT_GOAL_CTRL = {"T":t,"x":x,"y":y,"z":z,"t":tor,"spd":spd}
    create_json_commands(url, CMD_XYZT_GOAL_CTRL)
loc_x = 309.
loc_y = -5.
loc_z = 237.
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up subplots for x, y, z movement
fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(10, 6))
ax_x.set_title("X Position of Hand")
ax_y.set_title("Y Position of Hand")
ax_z.set_title("Z Position of Hand")

# Lists to store x, y, z coordinates
x_coords = []
y_coords = []
z_coords = []

# Set the initial position for transformation (350, -5, 237)
initial_position = (350, -5, 237)

def update_plot(frame):
    # Clear and update the plot each frame
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()

    ax_x.set_title("X Position of Hand")
    ax_y.set_title("Y Position of Hand")
    ax_z.set_title("Z Position of Hand")

    # Plot the x, y, z movement over time
    ax_x.plot(x_coords, label='X Position')
    ax_y.plot(y_coords, label='Y Position')
    ax_z.plot(z_coords, label='Z Position')

    ax_x.legend()
    ax_y.legend()
    ax_z.legend()

    ax_x.set_xlabel("Time (frames)")
    ax_y.set_xlabel("Time (frames)")
    ax_z.set_xlabel("Time (frames)")

    ax_x.set_ylabel("Position")
    ax_y.set_ylabel("Position")
    ax_z.set_ylabel("Position")

    plt.tight_layout()

# Set up the figure display and plot in the background (this keeps the plot updated)
plt.ion()  # Turn on interactive mode
plt.show()
count = 0
take_val=0
fist_open = 0

while cap.isOpened():
    if count == 4:
        take_val = 1
        count = 0
    count = count + 1

    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the hand landmarks
    hand_results = hands.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the image
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Access hand landmarks
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                #global smoothed_x, smoothed_y, smoothed_z
                smoothed_x = smooth(x, smoothed_x)
                smoothed_y = smooth(y, smoothed_y)
                smoothed_z = smooth(z, smoothed_z)
                # Transforming the coordinates based on initial position offset
                x_transformed = initial_position[0] -(smoothed_z * 1000)   # Map x to image width and apply transformation
                y_transformed = (smoothed_x * 640/2) + initial_position[1]  # Map y to image height and apply transformation
                z_transformed = (smoothed_y*10) + initial_position[2]  # Arbitrary scaling for z coordinate
                if z_transformed < -74.:
                    z_transformed = -70
                
                # Add transformed coordinates to the lists
                x_coords.append(x_transformed)
                y_coords.append(y_transformed)
                z_coords.append(z_transformed)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                

                # You can use the distance between thumb tip and wrist to detect if fist is open or closed
                distance = ((wrist.x - thumb_tip.x) ** 2 + (wrist.y - thumb_tip.y) ** 2 + (wrist.z - thumb_tip.z) ** 2) ** 0.5
                if distance < 0.05:  # Threshold for detecting a fist close
                    print("Fist is closed")
                    fist_open= 0
                else:
                    #print("Fist is open")
                    fist_open=1


                # Display the transformed coordinates on the console
                #print(f"Hand Landmark - x: {x_transformed:.2f}, y: {y_transformed:.2f}, z: {z_transformed:.2f}")
                if take_val:
                    
                    if (fist_open):
                        move_to_xyzt(my_url, 104, round(x_transformed, 4),round(y_transformed,4), loc_z, 3.14,10)
                        
                    else:
                        move_to_xyzt(my_url, 104, round(x_transformed,4),round(y_transformed, 4), loc_z, 3.14,10)
                        

                    take_val=0
                # Plot the transformed landmark on the image (in blue)
                height, width, _ = frame.shape
                cx, cy = int(x * width), int(y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a blue circle at each landmark

    # Show the webcam feed with landmarks
    cv2.imshow("Hand Tracking", frame)

    # Update the plot window
    update_plot(None)
    plt.pause(0.01)  # Short pause to allow the plot to update

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot