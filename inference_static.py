import os
import numpy as np
import cv2
import joblib
from collections import deque, defaultdict
import json
from ultralytics import YOLO
from PIL import Image

# Define the model
model = YOLO("models/yolov8m-pose.pt")

# Load the saved random forest model and scaler
loaded_model = joblib.load('models/random_forest_model.joblib')
loaded_scaler = joblib.load('models/scaler.joblib')

# Constants
KEYPOINT_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, 
    "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}
FALL_THRESHOLD = 10  # Number of consecutive frames to consider a fall
FRAGMENT_DURATION = 10  # Total duration of fall fragment in seconds

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.sum(v1 * v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    cos_angle = dot_product / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_fall_attributes(keypoints, boxes):
    attributes = []
    for keypoint, box in zip(keypoints, boxes):
        # Extract key body points
        left_shoulder = keypoint[KEYPOINT_INDEX['left_shoulder']]
        right_shoulder = keypoint[KEYPOINT_INDEX['right_shoulder']]
        left_hip = keypoint[KEYPOINT_INDEX['left_hip']]
        right_hip = keypoint[KEYPOINT_INDEX['right_hip']]
        left_ankle = keypoint[KEYPOINT_INDEX['left_ankle']]
        right_ankle = keypoint[KEYPOINT_INDEX['right_ankle']]
        left_knee = keypoint[KEYPOINT_INDEX['left_knee']]
        right_knee = keypoint[KEYPOINT_INDEX['right_knee']]
        
        # Extract bounding box dimensions
        x, y, width, height = box

        # Aspect Ratio
        aspect_ratio = width / (height + 1e-6)

        # Hip Angle (angle between hips and horizontal line)
        hip_angle = np.arctan2(np.abs(right_hip[1] - left_hip[1]), 
                               np.abs(right_hip[0] - left_hip[0]))

        # Shoulder Angle (angle between shoulders and horizontal line)
        shoulder_angle = np.arctan2(np.abs(right_shoulder[1] - left_shoulder[1]), 
                                    np.abs(right_shoulder[0] - left_shoulder[0]))

        # Centroid Difference (difference between upper and lower centroids)
        upper_centroid = (left_shoulder + right_shoulder) / 2
        lower_centroid = (left_hip + right_hip) / 2
        centroid_difference = np.abs(lower_centroid[1] - upper_centroid[1]) / (height + 1e-6)

        # Deflection Angle (angle between centroid line and vertical)
        deflection_angle = np.arctan2(np.abs(upper_centroid[0] - lower_centroid[0]), 
                                      np.abs(upper_centroid[1] - lower_centroid[1]))

        # Hip to Ankle Angle (smallest angle between hips and ankles)
        hip_to_ankle_angle_left = calculate_angle(left_hip, left_knee, left_ankle)
        hip_to_ankle_angle_right = calculate_angle(right_hip, right_knee, right_ankle)
        hip_to_ankle_angle = min(hip_to_ankle_angle_left, hip_to_ankle_angle_right)

        # Shoulder to Ankle Angle (smallest angle between shoulders and ankles)
        shoulder_to_ankle_angle_left = calculate_angle(left_shoulder, left_hip, left_ankle)
        shoulder_to_ankle_angle_right = calculate_angle(right_shoulder, right_hip, right_ankle)
        shoulder_to_ankle_angle = min(shoulder_to_ankle_angle_left, shoulder_to_ankle_angle_right)
        
        attributes.append([width, height, aspect_ratio, hip_angle, shoulder_angle, centroid_difference,
                           deflection_angle, hip_to_ankle_angle, shoulder_to_ankle_angle])
    
    return np.array(attributes)

def predict_fall(fall_attributes):
    # Preprocess the input features
    features_scaled = loaded_scaler.transform(fall_attributes)
    
    # Make prediction
    predictions = loaded_model.predict(features_scaled)
    probabilities = loaded_model.predict_proba(features_scaled)[:, 1]  # Probability of positive class
    
    return predictions, probabilities

# Process each frame from YOLOv8 results
def process_frame(result):
    keypoints, boxes = [], []
    if result.keypoints is not None and result.boxes is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(keypoints) > 0 and len(boxes) > 0:
            fall_attributes = calculate_fall_attributes(keypoints, boxes)
            predictions, probabilities = predict_fall(fall_attributes)
    return keypoints, boxes, predictions, probabilities

# Main function to process the video
def main():
    input_dir = 'input'
    video_name = os.listdir(input_dir)[0]
    video_path = os.path.join(input_dir, video_name)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Real-time window for display
    cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)

    # Frame processing
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 inference on the current frame
        results = model(frame)

        # Process the results to extract keypoints and fall predictions
        for result in results:
            frame_copy = frame.copy()
            keypoints, boxes, predictions, probabilities = process_frame(result)

            # Annotate frame with fall detection results
            for i, (box, prediction, probability) in enumerate(zip(boxes, predictions, probabilities)):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                label = f"Person {i} - {'Fall' if prediction == 1 else 'No Fall'}: {probability:.2f}"
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw skeleton keypoints
                for j, (x, y) in enumerate(keypoints[i]):
                    cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # Display the annotated frame
            cv2.imshow('Fall Detection', frame_copy)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()