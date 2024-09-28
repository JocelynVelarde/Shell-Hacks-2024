# Nose
# Left Eye
# Right Eye
# Left Ear
# Right Ear
# Left Shoulder
# Right Shoulder
# Left Elbow
# Right Elbow
# Left Wrist
# Right Wrist
# Left Hip
# Right Hip
# Left Knee
# Right Knee
# Left Ankle
# Right Ankle

import cv2
import numpy as np
from ultralytics import YOLO

def draw_keypoints(image, keypoints, confidence, selected_keypoints):
    for person_keypoints, person_confidence in zip(keypoints, confidence):
        for idx in selected_keypoints:
            x, y = person_keypoints[idx]
            conf = person_confidence[idx]
            if conf > 0.5:  # Only draw keypoints with confidence > 0.5
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(image, f"{idx}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

# Load the YOLOv8 model
model = YOLO("yolov8m-pose.pt")

# Open the video file or webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam or "path/to/your/video.mp4" for a video file

# Specify the keypoints you're interested in
# For example, let's track nose (0), left shoulder (5), right shoulder (6), left hip (11), right hip (12)
selected_keypoints = [0, 5, 6, 11, 12]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Get the keypoints
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        confidence = results[0].keypoints.conf.cpu().numpy()
        
        # Draw only the selected keypoints
        annotated_frame = draw_keypoints(annotated_frame, keypoints, confidence, selected_keypoints)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()