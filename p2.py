import os
import cv2
import numpy as np
import requests
from ultralytics import YOLO
import time
import gridfs
import pymongo
from PIL import Image
import datetime
import json
import urllib.parse

with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

# Load the YOLOv8 pose estimation model
model = YOLO("yolov8m-pose.pt")

# Define a list of keypoints you're interested in for fall detection
KEYPOINT_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, 
    "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

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
    falls_indices = []
    for i in range(len(fall_attributes['width'])):
        # Threshold conditions
        is_fall = (
            fall_attributes['centroid_difference'][i] > thresholds['centroid_diff'] or
            fall_attributes['hip_angle'][i] > thresholds['angle'] or
            fall_attributes['shoulder_angle'][i] > thresholds['angle'] or
            fall_attributes['aspect_ratio'][i] > thresholds['aspect_ratio']
        )
        falls.append(is_fall)
        if is_fall:
            falls_indices.append(i)
    return falls, falls_indices


def download_video_from_mongoDB(uri, database, collection, video_id, output_path):
    client = MongoClient(uri)
    db = client[database]
    fs = gridfs.GridFS(db, collection)
    
    video_file = fs.get(video_id)
    
    with open(output_path, "wb") as f:
        f.write(video_file.read())
        
    print(f"Video downloaded to: {output_path}")


def saveFrameAasImage(frame, timestamp):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_path = f"fall_{timestamp}.jpg"
    image.save(image_path)
    return image_path

# def upload_video_to_api(api_url):
    
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         print("Video uploaded successfully.")
#     else:
#         print("Error uploading video:", response.text)

# def download_video(filename):
#     # Define the API endpoint
#     api_endpoint = f"http://10.110.199.23:8000/download_video/{filename}"

#     # Make the GET request to the API
#     response = requests.get(api_endpoint)

#     # Check the response status code
#     if response.status_code == 200:
#         # Save the video file if the request was successful
#         with open(f"{filename}.mp4", "wb") as file:
#             file.write(response.content)
#         print(f"Video '{filename}.mp4' downloaded successfully!")
#     else:
#         # Print the error message if the request failed
#         print("Failed to download video:", response.json())

def list_files_in_gridfs(uri, database):
    client = MongoClient(uri)
    db = client[database]
    fs = gridfs.GridFS(db)
    
    files = fs.list()
    for file in files:
        print(file)
        

# def download_video_from_mongoDB(uri, database, collection ,video_id, output_path):
#     client = MongoClient(uri)
#     db = client[database]
#     fs = gridfs.GridFS(db, collection)
#     video_file = fs.get(video_id)
#     with open(output_path, "wb") as f:
#         f.write(video_file.read())
    
#     print(f"Video downloaded to: {output_path}")

def download_video_from_mongoDB(uri, selected_video):
    client = pymongo.MongoClient(uri)
    db = client["video_database"]
    fs = gridfs.GridFS(db)
    
    video_metadata = db.videos.find_one({"filename": selected_video})
    
    if video_metadata:
        file_id = video_metadata["file_id"]
        grid_out = fs.get(file_id)
        video_bytes = grid_out.read()
        with open(selected_video, "wb") as f:
            f.write(video_bytes)
        print(f"Video downloaded to: {selected_video}")
    else:
        print(f"Error: Video '{selected_video}' not found in MongoDB")


def upload_video_to_mongo(uri, database, collection, video_path):
    # Connect to MongoDB
    client = MongoClient(uri)
    db = client[database]
    fs = gridfs.GridFS(db, collection)
    
    # Upload the video file to MongoDB
    with open(video_path, "rb") as f:
        video_id = fs.put(f, filename=video_path)
    
    print(f"Video uploaded with ID: {video_id}")
    return video_id

def uploadToMongoDB(video_path, fall_times, bounding_box_coords, images):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["fall_detection"]
    fs = gridfs.GridFS(db)
    

    with open(video_path, "rb") as f:
        video_id = fs.put(f, filename="output.mp4")
    

    for i, timestamp in enumerate(fall_times):
        bbox = bounding_box_coords[i]
        image_path = images[i]
    
        with open(image_path, "rb") as f:
            image_id = fs.put(f, filename=image_path)
 
        db.falls.insert_one({
            "timestamp": timestamp,
            "bounding_box": bbox,
            "image_id": image_id,
            "video_id": video_id,
            "date": datetime.datetime.now()
        })

def processVideoWithIntegratedCamPath(outputPath):
    thresholds = {
        'centroid_diff': 0.5,  # Adjust this value based on the scale of the bounding box
        'angle': np.radians(45),  # 45 degrees
        'aspect_ratio': 2.0  # Aspect ratio indicating a lying posture
    }
    capture_video(5, "video.mp4")
    video_path = "./video.mp4"
    cap = cv2.VideoCapture("video.mp4")
    
    fall_times = []
    while cap.isOpened():
        success, frame = cap.read()

        time_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if success:
            results = model(frame)
            time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_time = time - time_start

            keypoints = results[0].keypoints.xy if results[0].keypoints else None
            bboxes = results[0].boxes.xywh if results[0].boxes else None

            if keypoints is not None and bboxes is not None:
                fall_attributes = calculate_fall_attributes(keypoints, bboxes, frame_time)
                
                falls, fall_indices = detect_fall(fall_attributes, thresholds)
                
                if fall_indices:
                    fall_times.append(time)
                    print(f"Fall detected at time: {time:.2f} seconds")

                annotated_frame = results[0].plot()

                for i, bbox in enumerate(bboxes):
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    color = (0, 0, 255) if falls[i] else (0, 255, 0)
                    label = "Fall Detected" if falls[i] else "Normal"
                    cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("Fall Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    
def processVideoWithFile(videoPath): # video, timestap si se cayo, coords bounding box, foto del timestamp que se cayo
    thresholds = {
        'centroid_diff': 0.5,  # Adjust this value based on the scale of the bounding box
        'angle': np.radians(45),  # 45 degrees
        'aspect_ratio': 2.0  # Aspect ratio indicating a lying posture
    }
    cap = cv2.VideoCapture(videoPath)
    
    fall_times = []
    bounding_box_coords = []
    images = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        success, frame = cap.read()

        time_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if success:
            results = model(frame)
            time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_time = time - time_start

            # Get keypoints and bounding boxes for each detected person
            keypoints = results[0].keypoints.xy if results[0].keypoints else None
            bboxes = results[0].boxes.xywh if results[0].boxes else None

            if keypoints is not None and bboxes is not None:
                # Calculate fall-related attributes for all people
                fall_attributes = calculate_fall_attributes(keypoints, bboxes, frame_time)
                
                falls, fall_indices = detect_fall(fall_attributes, thresholds)
                
                if fall_indices:
                    fall_times.append(time)
                    bounding_box_coords.append(bboxes[fall_indices[0]])
                    image_path = saveFrameAasImage(frame, time)
                    images.append(image_path)
                    print(f"Fall detected at time: {time:.2f} seconds")

                # Annotate the frame with the detection results (bounding boxes, keypoints, etc.)
                annotated_frame = results[0].plot()

                # Draw "Fall Detected" or "Normal" label
                for i, bbox in enumerate(bboxes):
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    color = (0, 0, 255) if falls[i] else (0, 255, 0)
                    label = "Fall Detected" if falls[i] else "Normal"
                    cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display the annotated frame with bounding boxes, keypoints, and fall detection results
            out.write(frame)
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

# Main loop for processing video frames
def main():
    uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    database = "shell_hacks"
    collection = "fall_classifer"
    
    # upload_video_to_mongo(uri, database, collection, "video.mp4")
    
    # download_video("heatmap_new")
    
    # processVideoWithFile("./heatmap_new.mp4.mp4")
    # list_files_in_gridfs(uri, database)
    
    # download_video_from_mongoDB(uri, "heatmap_new (11).mp4")
    
    processVideoWithFile("heatmap_new (11).mp4", "output.mp4")
    # # Define thresholds for fall detection (these values can be tuned)
    # thresholds = {
    #     'centroid_diff': 0.5,  # Adjust this value based on the scale of the bounding box
    #     'angle': np.radians(45),  # 45 degrees
    #     'aspect_ratio': 2.0  # Aspect ratio indicating a lying posture
    # }
    # capture_video(5, "video.mp4")
    # # Open the video file (replace with your video path)
    # video_path = "./video.mp4"
    # cap = cv2.VideoCapture("video.mp4")
    
    # fall_times = []
    # # get the time of the video
    # # Loop through the video frames
    # while cap.isOpened():
    #     # Read a frame from the video
    #     success, frame = cap.read()

    #     time_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
    #     if success:
    #         # Run YOLOv8 inference on the frame (pose detection)
    #         results = model(frame)
    #         time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    #         frame_time = time - time_start

    #         # Get keypoints and bounding boxes for each detected person
    #         keypoints = results[0].keypoints.xy if results[0].keypoints else None
    #         bboxes = results[0].boxes.xywh if results[0].boxes else None

    #         if keypoints is not None and bboxes is not None:
    #             # Calculate fall-related attributes for all people
    #             fall_attributes = calculate_fall_attributes(keypoints, bboxes, frame_time)
                
    #             falls, fall_indices = detect_fall(fall_attributes, thresholds)
                
    #             if fall_indices:
    #                 fall_times.append(time)
    #                 print(f"Fall detected at time: {time:.2f} seconds")

    #             # Annotate the frame with the detection results (bounding boxes, keypoints, etc.)
    #             annotated_frame = results[0].plot()

    #             # Draw "Fall Detected" or "Normal" label
    #             for i, bbox in enumerate(bboxes):
    #                 x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    #                 color = (0, 0, 255) if falls[i] else (0, 255, 0)
    #                 label = "Fall Detected" if falls[i] else "Normal"
    #                 cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    #         # Display the annotated frame with bounding boxes, keypoints, and fall detection results
    #         cv2.imshow("Fall Detection", annotated_frame)

    #         # Break the loop if 'q' is pressed
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break
    #     else:
    #         # Break the loop if the end of the video is reached
    #         break

    # # Release the video capture object and close the display window
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    database = "shell_hacks"
    collection = "fall_classifer"
    main()
    # capture_video(5, "output.mp4")
    
    
    # prompt form the time that the person fell 
    #find timestamp of the fall if detect fall true the person fall
    #from this timestamp make a clip
    #fit this clip to the llm model what cause the fall
    #input 