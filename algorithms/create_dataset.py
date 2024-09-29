import os
import pickle
import numpy as np
from ultralytics import YOLO
import cv2

# Define the model
model = YOLO("yolov8m-pose.pt")

# Define a list of keypoints you're interested in for fall detection
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

def calculate_fall_attributes(keypoints, boxes):
    keypoints = keypoints.cpu().numpy()[0]  # Get the first person's keypoints
    boxes = boxes.cpu().numpy()[0]  # Get the first person's bounding box
    
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
    x, y, width, height = boxes

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

def get_pose_embedding(model, image_path):
    image = cv2.imread(image_path)
    result = model(image)
    
    keypoints = result[0].keypoints.xy if result[0].keypoints else None
    bboxes = result[0].boxes.xywh if result[0].boxes else None

    if keypoints is not None and bboxes is not None:
        # Calculate fall-related attributes
        fall_attributes = calculate_fall_attributes(keypoints, bboxes)
    else:
        fall_attributes = np.array([None] * 9)
    
    return fall_attributes

def process_fall_case(fall_dir, output_folder):
    images = sorted([f for f in os.listdir(fall_dir) if f.endswith('.png')])
    labels = sorted([f for f in os.listdir(fall_dir) if f.endswith('.txt') and 'classes' not in f])

    data = []
    classes = []

    for image, label in zip(images, labels):
        image_path = os.path.join(fall_dir, image)
        label_path = os.path.join(fall_dir, label)

        # Get the pose embedding
        attr = get_pose_embedding(model, image_path)
        data.append(attr)

        # Read the label
        with open(label_path, 'r') as file:
            line = file.readline().strip()
            label = line.split()[0]
            classes.append(label)

    # Save the data into pkl file
    with open(os.path.join(output_folder, 'data.pkl'), 'wb') as file:
        pickle.dump(data, file)

    # Save the classes into pkl file
    with open(os.path.join(output_folder, 'classes.pkl'), 'wb') as file:
        pickle.dump(classes, file)

def main():
    raw_data_dir = 'Dataset CAUCAFall/CAUCAFall/'
    dataset_dir = 'dataset/'

    subjects = sorted(os.listdir(raw_data_dir))

    for i, subject in enumerate(subjects):
        subject_dir = os.path.join(raw_data_dir, subject)
        falls = sorted(os.listdir(subject_dir))

        for j, fall in enumerate(falls):
            falls_dir = os.path.join(subject_dir, fall)
            
            # Create the folder for the subject
            folder = os.path.join(dataset_dir, f'Subject{i+1}_Fall{j+1}')
            os.makedirs(folder, exist_ok=True)
            
            # Process fall case
            process_fall_case(falls_dir, folder)
            print(f"Processed: Subject{i+1}_Fall{j+1}")

if __name__ == "__main__":
    main()