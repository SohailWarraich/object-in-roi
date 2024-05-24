import cv2

# List to store clicked points
clicked_points = []

# Define the callback function to capture mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
        clicked_points.append((x, y))
        print("Clicked at (x, y) =", x, y)

# Path to the video file
video_path = './video3.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # Added WINDOW_NORMAL flag

# Set the mouse callback function for the window
cv2.setMouseCallback('Video', mouse_callback)

while True:
    # Read the first frame
    ret, frame = cap.read()
    
    # Break the loop if there are no more frames
    if not ret:
        break

    # Display the frame
    cv2.imshow('Video', frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Press 'n' to go to the next frame
            break
        elif key == 27:  # Press 'Esc' to exit
            cap.release()
            cv2.destroyAllWindows()
            print("Clicked points:", clicked_points)
            exit()

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print all clicked points
print("Clicked points:", clicked_points)
