import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model (adjust the path if needed)
model = YOLO('yolov8n.pt')

# Open the video file (modify the path to your video)
video_path = "./input_video/video2.mp4"
cap = cv2.VideoCapture(video_path)

# Define the ROI polygon (adjust points to fit your video)
region_points2 = [(11, 222), (347, 40), (588, 91), (432, 347), (24, 273)]

# Get the class names
class_names = model.names

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
output_path = './output_video/output_video2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'MJPG', etc.
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Define the font and colors
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (255, 255, 255)  # White
box_color =  (0, 127, 255)  # bounding box color
box_thickness = 1
background_color = (220, 0, 0) # Navy blue
roi_color = (57, 255, 20)  # Neon Green color

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Draw the ROI polygon
        cv2.polylines(frame, [np.array(region_points2, np.int32)], True, roi_color, 2)

        # Run YOLOv8 tracking on the frame
        results = model.predict(frame, classes=[0], conf=0.3, iou=0.3)
        person_count = 0

        for r in results:
            for box in r.boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = [int(x) for x in b]
                class_id = int(box.cls[0])  # Get the class id
                label = class_names[class_id]

                # Calculate the center point of the bounding box
                center_x = ((x1 + x2) / 2)
                center_y = ((y1 + y2) / 2)
                center_point = (int(center_x), int(center_y))

                # Check if the center point is inside the ROI
                result = cv2.pointPolygonTest(np.array(region_points2, dtype=np.int32), center_point, True)
                if result > 0:
                    person_count += 1

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness, cv2.LINE_AA)

                    # Draw the class label with background
                    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    text_x = x1
                    text_y = y1 - 10 if y1 > 20 else y1 + 20
                    cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + text_size[1] // 2), box_color, -1)
                    cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    # Draw the center point
                    cv2.circle(frame, center_point, 1, box_color, -1)

        # Draw the number of persons in the ROI on the top left corner of the frame
        person_text = f'Persons: {person_count}'
        text_size, _ = cv2.getTextSize(person_text, font, font_scale, font_thickness)
        text_x = 10
        text_y = 30
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0] + 20, text_y + 10), background_color, -1)
        cv2.putText(frame, person_text, (text_x + 10, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Write the frame to the output video file
        out.write(frame)

        # Display the frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()