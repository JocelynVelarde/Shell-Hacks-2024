import os
import numpy as np
from ultralytics import YOLO
import cv2
import joblib
from collections import deque, defaultdict
import time

# Define the model
model = YOLO("models/yolov8m-pose.pt")

# Load the saved model and scaler
loaded_model = joblib.load('models/random_forest_model.joblib')
loaded_scaler = joblib.load('models/scaler.joblib')

# Constants
KEYPOINT_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, 
    "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}
FALL_THRESHOLD = 10  # Number of consecutive frames to consider a fall
FRAGMENT_DURATION = 20  # Total duration of fall fragment in seconds
PRE_FALL_DURATION = FRAGMENT_DURATION // 2  # Duration to capture before the fall

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


def process_frame(frame):
    results = model(frame)
    
    annotated_frame = frame.copy()
    
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(keypoints) > 0 and len(boxes) > 0:
            fall_attributes = calculate_fall_attributes(keypoints, boxes)
            predictions, probabilities = predict_fall(fall_attributes)
            
            for i, (box, prediction, probability) in enumerate(zip(boxes, predictions, probabilities)):
                # Annotate the frame
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                label = f"Person {i} - {'Fall' if prediction == 1 else 'No Fall'}: {probability:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw skeleton
                for j, (x1, y1) in enumerate(keypoints[i]):
                    cv2.circle(annotated_frame, (int(x1), int(y1)), 5, (0, 255, 0), -1)
        
            return annotated_frame, predictions, probabilities, range(len(predictions))  # Return person IDs
    
    return annotated_frame, [], [], []  # Return annotated_frame, empty predictions, probabilities, and person IDs when no person is detected

def save_fall_fragment(pre_fall_frames, post_fall_frames, person_id):
    os.makedirs("fall_event_videos", exist_ok=True)
    output_path = f"fall_event_videos/person_{person_id}_fall_event.mp4"
    
    all_frames = pre_fall_frames + post_fall_frames
    
    if len(all_frames) > 0:
        height, width, _ = all_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame in all_frames:
            out.write(frame)
        
        out.release()
        print(f"Saved fall event video for Person {person_id} at {output_path}")
        print(f"Pre-fall frames: {len(pre_fall_frames)}, Post-fall frames: {len(post_fall_frames)}")
    else:
        print(f"No frames to save for Person {person_id}")

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    pre_fall_buffer_size = int(fps * PRE_FALL_DURATION)
    
    pre_fall_buffer = deque(maxlen=pre_fall_buffer_size)
    fall_counters = defaultdict(int)
    fall_detected = defaultdict(bool)
    post_fall_frames = defaultdict(list)
    post_fall_counter = defaultdict(int)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        
        pre_fall_buffer.append(frame)
        
        annotated_frame, predictions, probabilities, person_ids = process_frame(frame)
        
        # Check for falls for each person
        for person_id, prediction in zip(person_ids, predictions):
            if prediction == 1:  # Fall detected
                fall_counters[person_id] += 1
                if fall_counters[person_id] >= FALL_THRESHOLD and not fall_detected[person_id]:
                    fall_detected[person_id] = True
                    print(f"Fall detected for Person {person_id}! Starting to capture post-fall frames.")
                elif fall_detected[person_id]:
                    post_fall_frames[person_id].append(frame)
                    post_fall_counter[person_id] += 1
                    if post_fall_counter[person_id] >= pre_fall_buffer_size:
                        print(f"Completed capturing frames for Person {person_id}. Saving video fragment.")
                        save_fall_fragment(list(pre_fall_buffer), post_fall_frames[person_id], person_id)
                        fall_detected[person_id] = False
                        post_fall_frames[person_id] = []
                        post_fall_counter[person_id] = 0
            else:
                fall_counters[person_id] = 0
                if fall_detected[person_id]:
                    post_fall_frames[person_id].append(frame)
                    post_fall_counter[person_id] += 1
                    if post_fall_counter[person_id] >= pre_fall_buffer_size:
                        print(f"Completed capturing frames for Person {person_id}. Saving video fragment.")
                        save_fall_fragment(list(pre_fall_buffer), post_fall_frames[person_id], person_id)
                        fall_detected[person_id] = False
                        post_fall_frames[person_id] = []
                        post_fall_counter[person_id] = 0
        
        cv2.imshow("Fall Detection", annotated_frame)
        
        for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
            print(f"Person {i} - Prediction: {'Fall' if prediction == 1 else 'No Fall'}")
            print(f"Person {i} - Probability of Fall: {probability:.4f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()