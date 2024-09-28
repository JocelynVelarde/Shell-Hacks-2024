import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 pose estimation model
model = YOLO("yolov8m-pose.pt")

# Open the video file (replace with your video path)
video_path = "https://www.youtube.com/watch?v=yk6UVnMn9ts&ab_channel=Apple"
cap = cv2.VideoCapture(video_path)

# Define a list of keypoints you're interested in for fall detection
KEYPOINT_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, 
    "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

# Utility function to calculate the angle between three points (vectorized)
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.sum(v1 * v2, axis=1)
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)
    
    cos_angle = dot_product / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)

# Function to calculate attributes for fall detection (vectorized)
def calculate_fall_attributes(keypoints, boxes, frame_time):
    keypoints = keypoints.cpu().numpy()
    boxes = boxes.cpu().numpy()
    
    # Extract key body points
    left_shoulder = keypoints[:, KEYPOINT_INDEX['left_shoulder']]
    right_shoulder = keypoints[:, KEYPOINT_INDEX['right_shoulder']]
    left_hip = keypoints[:, KEYPOINT_INDEX['left_hip']]
    right_hip = keypoints[:, KEYPOINT_INDEX['right_hip']]
    left_ankle = keypoints[:, KEYPOINT_INDEX['left_ankle']]
    right_ankle = keypoints[:, KEYPOINT_INDEX['right_ankle']]
    left_knee = keypoints[:, KEYPOINT_INDEX['left_knee']]
    right_knee = keypoints[:, KEYPOINT_INDEX['right_knee']]
    nose = keypoints[:, KEYPOINT_INDEX['nose']]
    left_wrist = keypoints[:, KEYPOINT_INDEX['left_wrist']]
    right_wrist = keypoints[:, KEYPOINT_INDEX['right_wrist']]
    left_elbow = keypoints[:, KEYPOINT_INDEX['left_elbow']]
    right_elbow = keypoints[:, KEYPOINT_INDEX['right_elbow']]
    left_eye = keypoints[:, KEYPOINT_INDEX['left_eye']]
    right_eye = keypoints[:, KEYPOINT_INDEX['right_eye']]
    left_ear = keypoints[:, KEYPOINT_INDEX['left_ear']]
    right_ear = keypoints[:, KEYPOINT_INDEX['right_ear']]
    
    # Extract bounding box dimensions
    x, y, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Aspect Ratio
    aspect_ratio = width / (height + 1e-6)

    # Hip Angle (angle between hips and horizontal line)
    hip_angle = np.arctan2(np.abs(right_hip[:, 1] - left_hip[:, 1]), 
                           np.abs(right_hip[:, 0] - left_hip[:, 0]))

    # Shoulder Angle (angle between shoulders and horizontal line)
    shoulder_angle = np.arctan2(np.abs(right_shoulder[:, 1] - left_shoulder[:, 1]), 
                                np.abs(right_shoulder[:, 0] - left_shoulder[:, 0]))

    # Centroid Difference (difference between upper and lower centroids)
    upper_centroid = (left_shoulder + right_shoulder) / 2
    lower_centroid = (left_hip + right_hip) / 2
    centroid_difference = np.abs(lower_centroid[:, 1] - upper_centroid[:, 1]) / (height + 1e-6)

    # Deflection Angle (angle between centroid line and vertical)
    deflection_angle = np.arctan2(np.abs(upper_centroid[:, 0] - lower_centroid[:, 0]), 
                                  np.abs(upper_centroid[:, 1] - lower_centroid[:, 1]))

    # Hip to Ankle Angle (smallest angle between hips and ankles)
    hip_to_ankle_angle_left = calculate_angle(left_hip, left_knee, left_ankle)
    hip_to_ankle_angle_right = calculate_angle(right_hip, right_knee, right_ankle)
    hip_to_ankle_angle = np.minimum(hip_to_ankle_angle_left, hip_to_ankle_angle_right)

    # Shoulder to Ankle Angle (smallest angle between shoulders and ankles)
    shoulder_to_ankle_angle_left = calculate_angle(left_shoulder, left_hip, left_ankle)
    shoulder_to_ankle_angle_right = calculate_angle(right_shoulder, right_hip, right_ankle)
    shoulder_to_ankle_angle = np.minimum(shoulder_to_ankle_angle_left, shoulder_to_ankle_angle_right)
    
    # To track centroid vertical velocity, you need the previous frame's centroid height.
    # Define a variable to store the previous frame's centroid height
    previous_centroid_height = np.zeros(len(boxes))

    # Calculate vertical velocity (difference in centroid height between consecutive frames)
    vertical_velocity = (upper_centroid - previous_centroid_height) / frame_time  # frame_time is the time between frames
    previous_centroid_height = upper_centroid  # Update for the next frame

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "hip_angle": hip_angle,
        "shoulder_angle": shoulder_angle,
        "centroid_difference": centroid_difference,
        "deflection_angle": deflection_angle,
        "hip_to_ankle_angle": hip_to_ankle_angle,
        "shoulder_to_ankle_angle": shoulder_to_ankle_angle,
        "vertical_velocity": vertical_velocity
    }

# Function to detect falls based on thresholds
def detect_fall(fall_attributes, thresholds):
    falls = []
    for i in range(len(fall_attributes['width'])):
        # Threshold conditions
        is_fall = (
            fall_attributes['centroid_difference'][i] > thresholds['centroid_diff'] or
            fall_attributes['hip_angle'][i] > thresholds['angle'] or
            fall_attributes['shoulder_angle'][i] > thresholds['angle'] or
            fall_attributes['aspect_ratio'][i] > thresholds['aspect_ratio']
        )
        falls.append(is_fall)
    return falls

# Main loop for processing video frames
def main():
    # Define thresholds for fall detection (these values can be tuned)
    thresholds = {
        'centroid_diff': 0.5,  # Adjust this value based on the scale of the bounding box
        'angle': np.radians(45),  # 45 degrees
        'aspect_ratio': 2.0  # Aspect ratio indicating a lying posture
    }
    video_path = "https://www.youtube.com/watch?v=yk6UVnMn9ts&ab_channel=Apple"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        time_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame (pose detection)
            results = model(frame)
            time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_time = time - time_start

            # Get keypoints and bounding boxes for each detected person
            keypoints = results[0].keypoints.xy if results[0].keypoints else None
            bboxes = results[0].boxes.xywh if results[0].boxes else None

            if keypoints is not None and bboxes is not None:
                # Calculate fall-related attributes for all people
                fall_attributes = calculate_fall_attributes(keypoints, bboxes, frame_time)

                # Detect falls based on attributes
                falls = detect_fall(fall_attributes, thresholds)

                # Annotate the frame with the detection results (bounding boxes, keypoints, etc.)
                annotated_frame = results[0].plot()

                # Draw "Fall Detected" or "Normal" label
                for i, bbox in enumerate(bboxes):
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    color = (0, 0, 255) if falls[i] else (0, 255, 0)
                    label = "Fall Detected" if falls[i] else "Normal"
                    cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display the annotated frame with bounding boxes, keypoints, and fall detection results
            cv2.imshow("Fall Detection", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()