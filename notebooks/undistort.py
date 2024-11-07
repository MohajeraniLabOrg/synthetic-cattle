import cv2
import numpy as np

# Load your distortion coefficients (usually from a calibration process)
camera_matrix = np.array([[1.22e3, 0, 9.60e2], 
                          [0, 1.22e3, 5.40e2], 
                          [0, 0, 1]])

dist_coeffs = np.array([-0.2, 0, 0, 0, 0])

# Open the video file
file_name = 29
cap = cv2.VideoCapture(f'{file_name}.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output video
out = cv2.VideoWriter(f'{file_name}-hd.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

# Read the first frame
ret, frame = cap.read()
if ret:
    # Undistort the first frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    undistorted_frame = cv2.resize(undistorted_frame, (1280, 720))

    # Show the first undistorted frame
    cv2.imshow('First Undistorted Frame', undistorted_frame)

    # Wait for user input
    key = cv2.waitKey(0)
    if key == ord('q'):  # Quit if 'q' is pressed
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        exit()  # Exit the script

    # Proceed if Enter is pressed
    cv2.destroyAllWindows()

# Process the rest of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    undistorted_frame = cv2.resize(undistorted_frame, (1280, 720))

    # Write the undistorted frame to the output video
    out.write(undistorted_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
