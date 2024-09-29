import os
import numpy as np
import cv2
import joblib
from collections import deque, defaultdict
import json
from ultralytics import YOLO
from PIL import Image

# Define the model
model = YOLO("yolov8m-pose.pt")

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

def process_frame(result):
    keypoints, boxes = [], []
    if result.keypoints is not None and result.boxes is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(keypoints) > 0 and len(boxes) > 0:
            fall_attributes = calculate_fall_attributes(keypoints, boxes)
            predictions, probabilities = predict_fall(fall_attributes)
    return keypoints, boxes, predictions, probabilities
        
def main():
    # Load input video
    input_dir = 'input'
    output_dir = 'output'
    video_name = os.listdir(input_dir)[0]
    video_path = os.path.join(input_dir, video_name)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_dir, 'output_' + video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # For recording bounding boxes, keypoints, and falls
    bbox_data = []
    keypoints_data = []
    fall_timestamps = []
    
    # Tracking falls
    fall_count = 0
    fall_frames = deque(maxlen=FALL_THRESHOLD)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = model(frame)
        for result in results:
            keypoints, boxes, predictions, probabilities = process_frame(result)
            
            # Draw bounding boxes and keypoints on the frame
            for box, keypoint, prediction in zip(boxes, keypoints, predictions):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw keypoints
                for point in keypoint:
                    x, y = map(int, point)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
                # Add bounding box and keypoint data
                bbox_data.append({"frame": frame_idx, "bbox": [x1, y1, x2, y2]})
                keypoints_data.append({"frame": frame_idx, "keypoints": keypoint.tolist()})
                
                # Check for falls and mark it on the frame
                if prediction == 1:  # Fall detected
                    fall_count += 1
                    cv2.putText(frame, "Fall Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    fall_frames.append(1)
                else:
                    fall_frames.append(0)
            
            # Detect a fall if FALL_THRESHOLD consecutive frames have detected a fall
            if sum(fall_frames) >= FALL_THRESHOLD:
                timestamp = frame_idx / fps
                fall_timestamps.append(timestamp)
        
        # Write processed frame to output video
        out.write(frame)
        frame_idx += 1

    # Save bounding box, keypoint, and fall data
    output_json = os.path.join(output_dir, 'data_' + video_name.replace('.mp4', '.json'))
    with open(output_json, 'w') as f:
        json.dump({"bboxes": bbox_data, "keypoints": keypoints_data, "fall_timestamps": fall_timestamps}, f)
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()