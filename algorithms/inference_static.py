import os
import numpy as np
import cv2
import joblib
from collections import deque, defaultdict
import json
from ultralytics import YOLO
from PIL import Image
import time
import pymongo
import gridfs
import urllib.parse
import torch.nn as nn
import torch

# Define the model
class FallDetectionModel(nn.Module):
    def _init_(self, input_size):
        super(FallDetectionModel, self)._init_()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# Define the model
model = YOLO("models/yolov8m-pose.pt")

# Load the saved random forest model and scaler
loaded_model = joblib.load('models/random_forest_model.joblib')
loaded_scaler = joblib.load('models/scaler.joblib')

# Load the MLP:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = FallDetectionModel(9).to(device)
mlp.load_state_dict(torch.load('models/fall_detection_model.pth'))
mlp.eval()

with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Constants
KEYPOINT_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, 
    "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}
CONSECUTIVE_FALL_FRAMES = 10
FRAGMENT_DURATION = 10  # Total duration of fall fragment in seconds

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.sum(v1 * v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    cos_angle = dot_product / (v1_norm * v2_norm + 1e-6)
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

def predict_fall(fall_attributes, algorithm='ml'):
    if algorithm == 'ml':
        # Preprocess the input features
        features_scaled = loaded_scaler.transform(fall_attributes)
        
        # Make prediction
        predictions = abs(1 - loaded_model.predict(features_scaled))  # Invert the prediction (0: Fall, 1: Stable)
        probabilities = loaded_model.predict_proba(features_scaled)[:, 1]  # Probability of positive class
    elif algorithm == 'dl':
        fall_attributes_tensor = torch.FloatTensor(fall_attributes).to(device)
        probabilities = mlp(fall_attributes_tensor)
        predictions = (probabilities > 0.5).int().cpu().numpy()
    return predictions, probabilities

# Dictionary to hold fall events
fall_events = []

# Initialize a counter for consecutive fall detections
consecutive_fall_count = 0
fall_detected = False

# Modify the process_frame function to track falls
def process_frame(result, frame_index, frame_time):
    global consecutive_fall_count, fall_detected
    
    keypoints, boxes, scores = [], [], []
    if result.keypoints is not None and result.boxes is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()  # Extract confidence scores

        if len(keypoints) > 0 and len(boxes) > 0:
            fall_attributes = calculate_fall_attributes(keypoints, boxes)
            predictions, probabilities = predict_fall(fall_attributes)
            
            # Check if fall is detected
            for i, prediction in enumerate(predictions):
                if prediction == 1:
                    consecutive_fall_count += 1
                    if consecutive_fall_count == CONSECUTIVE_FALL_FRAMES:
                        # Log fall event
                        fall_events.append({
                            "timestamp": frame_time,
                            "box": boxes[i].tolist(),
                            "keypoints": keypoints[i].tolist()
                        })
                        fall_detected = True
                else:
                    consecutive_fall_count = 0  # Reset counter if no fall is detected
        else:
            # Reset fall detection if no keypoints/boxes are detected
            consecutive_fall_count = 0

    return keypoints, boxes, predictions, probabilities, scores  # Return confidence scores as well


def capture_video(duration, output_path):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default to 30 if unable to get FPS
    
    # Calculate the total number of frames to capture
    total_frames = int(duration * fps)
    
    # Get the width and height of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    start_time = time.time()
    frame_count = 0
    while frame_count < total_frames:
        success, frame = cap.read()
        if not success:
            break
        
        # Write the frame to the video file
        out.write(frame)
        
        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Release the camera and video writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved to: {output_path}")

def download_video_from_mongoDB(uri, selected_video):
    client = pymongo.MongoClient(uri)
    db = client["video_database"]
    fs = gridfs.GridFS(db)
    
    video_metadata = db.videos.find_one({"filename": selected_video})
    
    if video_metadata:
        file_id = video_metadata["file_id"]
        grid_out = fs.get(file_id)
        video_bytes = grid_out.read()
        with open(f"./input/{selected_video}", "wb") as f:
            f.write(video_bytes)
        print(f"Video downloaded to: {selected_video}")
    else:
        print(f"Error: Video '{selected_video}' not found in MongoDB")
        
def process_video(video_path):
    # input_dir = 'input'
    # video_name = os.listdir(input_dir)[0]
    # video_path = os.path.join(input_dir, video_name)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video details for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Output path for the saved video with annotations
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'output_with_fall_detection.avi')

    # Create VideoWriter to save annotated video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Real-time window for display
    cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)

   # Frame processing with timestamp
    frame_time = 0  # Initialize frame time counter
    frame_index = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 inference on the current frame
        results = model(frame)

        # Process the results to extract keypoints and fall predictions
        for result in results:
            frame_copy = frame.copy()

            # Process the current frame with timestamp
            keypoints, boxes, predictions, probabilities, scores = process_frame(result, frame_index, frame_time)

            # Annotate frame with fall detection results
            for i, (box, prediction, probability, score) in enumerate(zip(boxes, predictions, probabilities, scores)):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                label1 = f"Person {i} - {'Fall' if prediction == 1 else 'Stable'}: {probability:.2f}"
                label2 = f"Confidence: {score:.2f}"
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_copy, label1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame_copy, label2, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw skeleton keypoints
                for j, (x, y) in enumerate(keypoints[i]):
                    cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # Write annotated frame to video file
            out.write(frame_copy)
            
            # Display the annotated frame
            cv2.imshow('Fall Detection', frame_copy)
            
            # Increment frame time based on FPS
            frame_time += 1 / fps
            frame_index += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
    return output_video_path

    # Cleanup
    cap.release()
    out.release()  # Ensure the video writer is closed properly
    cv2.destroyAllWindows()
    
    # Save fall events to a JSON file
    with open(os.path.join(output_dir, 'fall_events.json'), 'w') as f:
        json.dump(fall_events, f)

    print(f"Annotated video saved to: {output_video_path}")
    print(f"Fall events saved to: fall_events.json")

def upload_json_to_mongoDB(uri, json_path):
    client = pymongo.MongoClient(uri)
    db = client["video_database"]
    fs = gridfs.GridFS(db)
    
    with open(json_path, "r") as f:
        json_bytes = f.read()
    
    json_metadata = {
        "filename": os.path.basename(json_path),
        "file_id": fs.put(json_bytes)
    }
    
    db.jsons.insert_one(json_metadata)
    
    print(f"JSON uploaded to MongoDB: {json_path}")
    
def upload_video_to_mongoDB(uri, video_path):
    client = pymongo.MongoClient(uri)
    db = client["video_database"]
    fs = gridfs.GridFS(db)
    
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    video_metadata = {
        "filename": os.path.basename(video_path),
        "file_id": fs.put(video_bytes)
    }
    
    db.videos.insert_one(video_metadata)
    
    print(f"Video uploaded to MongoDB: {video_path}")
    

# Main function to process the video and save the output
def main():
    
    # download_video_from_mongoDB(uri, selected_video)
    
    # upload_video_to_mongoDB(uri, "input/video.mp4")
    
    
    
    process_video("input/input1.avi")
    

if __name__ == "__main__":
    
    main()