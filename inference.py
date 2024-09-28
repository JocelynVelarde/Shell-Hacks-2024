import os
import numpy as np
from ultralytics import YOLO
import cv2
import joblib

# Define the model
model = YOLO("models/yolov8m-pose.pt")

# Load the saved model and scaler
loaded_model = joblib.load('models/random_forest_model.joblib')
loaded_scaler = joblib.load('models/scaler.joblib')

# Define keypoints
KEYPOINT_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, 
    "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.sum(v1 * v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    cos_angle = dot_product / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_fall_attributes(keypoints, box):
    # Extract key body points
    left_shoulder = keypoints[KEYPOINT_INDEX['left_shoulder']]
    right_shoulder = keypoints[KEYPOINT_INDEX['right_shoulder']]
    left_hip = keypoints[KEYPOINT_INDEX['left_hip']]
    right_hip = keypoints[KEYPOINT_INDEX['right_hip']]
    left_ankle = keypoints[KEYPOINT_INDEX['left_ankle']]
    right_ankle = keypoints[KEYPOINT_INDEX['right_ankle']]
    left_knee = keypoints[KEYPOINT_INDEX['left_knee']]
    right_knee = keypoints[KEYPOINT_INDEX['right_knee']]
    
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

    return np.array([width, height, aspect_ratio, hip_angle, shoulder_angle, centroid_difference,
                     deflection_angle, hip_to_ankle_angle, shoulder_to_ankle_angle])

def predict_fall(fall_attributes):
    # Preprocess the input features
    features_scaled = loaded_scaler.transform(fall_attributes.reshape(1, -1))
    
    # Make prediction
    prediction = loaded_model.predict(features_scaled)
    probability = loaded_model.predict_proba(features_scaled)[0][1]  # Probability of positive class
    
    return prediction[0], probability

def process_frame(frame):
    results = model(frame)
    
    annotated_frame = frame.copy()
    
    for result in results:
        if result.keypoints is not None and result.boxes is not None:
            keypoints = result.keypoints.xy[0].cpu().numpy()
            box = result.boxes.xyxy[0].cpu().numpy()
            
            fall_attributes = calculate_fall_attributes(keypoints, box)
            prediction, probability = predict_fall(fall_attributes)
            
            # Annotate the frame
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
            label = f"{'Fall' if prediction == 1 else 'No Fall'}: {probability:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw skeleton
            for i, (x1, y1) in enumerate(keypoints):
                cv2.circle(annotated_frame, (int(x1), int(y1)), 5, (0, 255, 0), -1)
    
    return annotated_frame, prediction, probability

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        
        annotated_frame, prediction, probability = process_frame(frame)
        
        cv2.imshow("Fall Detection", annotated_frame)
        
        print(f"Prediction: {'Fall' if prediction == 1 else 'No Fall'}")
        print(f"Probability of Fall: {probability:.4f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()