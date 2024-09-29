import os
import numpy as np
import cv2
import joblib
from collections import deque, defaultdict
import json
from ultralytics import YOLO
from pytube import YouTube

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

def download_youtube_video(url, output_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        stream.download(output_path=output_path, filename='youtube_video.mp4')
        return os.path.join(output_path, 'youtube_video.mp4')
    except Exception as e:
        print(f"An error occurred while downloading the YouTube video: {e}")
        return None

def process_frame(frame, frame_number):
    results = model(frame)
    
    annotated_frame = frame.copy()
    bboxes = []
    keypoints = []
    
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(keypoints) > 0 and len(boxes) > 0:
            fall_attributes = calculate_fall_attributes(keypoints, boxes)
            predictions, probabilities = predict_fall(fall_attributes)
            
            for i, (box, prediction, probability) in enumerate(zip(boxes, predictions, probabilities)):
                x1, y1, x2, y2 = map(int, box)
                bboxes.append([frame_number, i, x1, y1, x2, y2])
                
                # Annotate the frame
                color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                label = f"Person {i} - {'Fall' if prediction == 1 else 'No Fall'}: {probability:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw skeleton
                for j, (x1, y1) in enumerate(keypoints[i]):
                    cv2.circle(annotated_frame, (int(x1), int(y1)), 5, (0, 255, 0), -1)
        
            return annotated_frame, predictions, probabilities, range(len(predictions)), bboxes, keypoints
    
    return annotated_frame, [], [], [], bboxes, keypoints

def save_output_video(frames, output_path):
    if len(frames) > 0:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Saved output video at {output_path}")
    else:
        print("No frames to save")

def save_json_data(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Saved JSON data at {output_path}")

def main():
    os.makedirs("output", exist_ok=True)
    video_path = 'test/Watch Your Step! Funny Slips and Falls Compilation _ FailArmy.mp4'

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_frames = []
    all_bboxes = []
    all_keypoints = []
    fall_events = []
    fall_counters = defaultdict(int)
    fall_detected = defaultdict(bool)
    
    frame_buffer = deque(maxlen=fps * FRAGMENT_DURATION)
    
    for frame_number in range(total_frames):
        success, frame = cap.read()
        if not success:
            break
        
        frame_buffer.append(frame)
        
        annotated_frame, predictions, probabilities, person_ids, bboxes, keypoints = process_frame(frame, frame_number)
        
        output_frames.append(annotated_frame)
        all_bboxes.extend(bboxes)
        all_keypoints.extend([{'frame': frame_number, 'person': i, 'keypoints': kp.tolist()} for i, kp in enumerate(keypoints)])
        
        # Check for falls for each person
        for person_id, prediction in zip(person_ids, predictions):
            if prediction == 1:  # Fall detected
                fall_counters[person_id] += 1
                if fall_counters[person_id] >= FALL_THRESHOLD and not fall_detected[person_id]:
                    fall_detected[person_id] = True
                    fall_frame = frame_number - FALL_THRESHOLD // 2
                    fall_events.append({'person': person_id, 'frame': fall_frame, 'time': fall_frame / fps})
                    print(f"Fall detected for Person {person_id} at frame {fall_frame}")
            else:
                fall_counters[person_id] = 0
        
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames")
    
    cap.release()
    
    # Save outputs
    print("Saving outputs...")
    
    # 1. Save annotated video
    save_output_video(output_frames, "output/annotated_video.mp4")
    
    # 2. Save bounding box coordinates
    save_json_data(all_bboxes, "output/bboxes.json")
    
    # 3. Save keypoint coordinates
    save_json_data(all_keypoints, "output/keypoints.json")
    
    # 4. Save fall events
    save_json_data(fall_events, "output/fall_events.json")
    
    # 5. Save 10-second clips for each fall event
    for i, event in enumerate(fall_events):
        start_frame = max(0, event['frame'] - fps * FRAGMENT_DURATION // 2)
        end_frame = min(total_frames, start_frame + fps * FRAGMENT_DURATION)
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        clip_frames = []
        for _ in range(end_frame - start_frame):
            success, frame = cap.read()
            if not success:
                break
            clip_frames.append(frame)
        
        cap.release()
        
        save_output_video(clip_frames, f"output/fall_event_clip_{i}.mp4")
    
    print("Processing complete. All outputs saved in the 'output' directory.")

if __name__ == "__main__":
    main()