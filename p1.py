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

# Load the YOLO pose estimation model
model = YOLO("models/yolov8m-pose.pt")

# Define a list of keypoints you're interested in for fall detection
# Assuming index order from the paper
KEYPOINT_INDEX = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12, 
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

# Utility function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    return np.degrees(angle)

# Function to calculate attributes for fall detection
def calculate_fall_attributes(keypoints):
    # Extract key body points
    left_shoulder = keypoints[KEYPOINT_INDEX['left_shoulder']]
    right_shoulder = keypoints[KEYPOINT_INDEX['right_shoulder']]
    left_hip = keypoints[KEYPOINT_INDEX['left_hip']]
    right_hip = keypoints[KEYPOINT_INDEX['right_hip']]
    left_ankle = keypoints[KEYPOINT_INDEX['left_ankle']]
    right_ankle = keypoints[KEYPOINT_INDEX['right_ankle']]
    left_knee = keypoints[KEYPOINT_INDEX['left_knee']]
    right_knee = keypoints[KEYPOINT_INDEX['right_knee']]
    
    # 1. Width (horizontal distance between farthest x coordinates)
    width = max(keypoints[:, 0]) - min(keypoints[:, 0])

    # 2. Height (vertical distance between farthest y coordinates)
    height = max(keypoints[:, 1]) - min(keypoints[:, 1])

    # 3. Aspect Ratio
    aspect_ratio = width / height if height != 0 else 0

    # 4. Hip Angle (angle between hips and horizontal line)
    hip_angle = np.arctan2(abs(right_hip[1] - left_hip[1]), abs(right_hip[0] - left_hip[0]))

    # 5. Shoulder Angle (angle between shoulders and horizontal line)
    shoulder_angle = np.arctan2(abs(right_shoulder[1] - left_shoulder[1]), abs(right_shoulder[0] - left_shoulder[0]))

    # 6. Centroid Difference (difference between upper and lower centroids)
    upper_centroid = np.mean([left_shoulder, right_shoulder], axis=0)
    lower_centroid = np.mean([left_hip, right_hip], axis=0)
    centroid_difference = abs(lower_centroid[1] - upper_centroid[1]) / height if height != 0 else 0

    # 7. Deflection Angle (angle between centroid line and vertical)
    deflection_angle = np.arctan2(abs(upper_centroid[0] - lower_centroid[0]), abs(upper_centroid[1] - lower_centroid[1]))

    # 8. Hip to Ankle Angle (smallest angle between hips and ankles)
    hip_to_ankle_angle_left = calculate_angle(left_hip, left_knee, left_ankle)
    hip_to_ankle_angle_right = calculate_angle(right_hip, right_knee, right_ankle)
    hip_to_ankle_angle = min(hip_to_ankle_angle_left, hip_to_ankle_angle_right)

    # 9. Shoulder to Ankle Angle (smallest angle between shoulders and ankles)
    shoulder_to_ankle_angle_left = calculate_angle(left_shoulder, left_hip, left_ankle)
    shoulder_to_ankle_angle_right = calculate_angle(right_shoulder, right_hip, right_ankle)
    shoulder_to_ankle_angle = min(shoulder_to_ankle_angle_left, shoulder_to_ankle_angle_right)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "hip_angle": hip_angle,
        "shoulder_angle": shoulder_angle,
        "centroid_difference": centroid_difference,
        "deflection_angle": deflection_angle,
        "hip_to_ankle_angle": hip_to_ankle_angle,
        "shoulder_to_ankle_angle": shoulder_to_ankle_angle
    }

def main():
    # Open the video source (camera)
    results = model(source=0, conf=0.3, save=True, stream=True)

    while True:
        for result in results:
            frame = result.orig_img  # Get the original image frame
            keypoints = result.keypoints.xy if result.keypoints else None
            
            if keypoints is not None:
                # Calculate fall-related attributes
                fall_attributes = calculate_fall_attributes(keypoints)
                print(fall_attributes)  # Output attributes to monitor them

            # Display the frame
            cv2.imshow('Fall Detection Frame', frame)

            # Break loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video source and close windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()