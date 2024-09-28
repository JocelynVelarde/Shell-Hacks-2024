import os
import pandas as pd
from ultralytics import YOLO
import cv2
from pytube import YouTube

model = YOLO("yolov8m-pose.pt")

    

def get_bounding_box_corners(track_box):
    x1, y1, x2, y2 = map(int, track_box)
    
    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)
    
    return top_left, top_right, bottom_left, bottom_right

def processVideo(videoPath, outputPath):
    input_video = cv2.VideoCapture(videoPath)
    
    if not input_video.isOpened():
        print(f"Can't open the video: {videoPath}")
        return
    
    frame_width = int(input_video.get(3))
    frame_height = int(input_video.get(4))
    
    outputVideo = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    results = model.track(videoPath, classes=0, show = True, conf=0.55, iou=0.6, tracker='./botsort.yaml')
    
    for frame_idx, result in enumerate(results):
        frame = result.plot()
        
        if frame is None:
            break
        
        if result.boxes.is_track:
            track_ids = result.boxes.id.int().cpu().tolist()
            track_boxes = result.boxes.xyxy.cpu().tolist()
            
            for track_id, track_box in zip(track_ids, track_boxes):
                top_left, top_right, bottom_left, bottom_right = get_bounding_box_corners(track_box)
                print(f"Track ID: {track_id}")
                print(f"Top Left: {top_left}, Top Right: {top_right}, Bottom Left: {bottom_left}, Bottom Right: {bottom_right}")
                
        outputVideo.write(frame)
        
    input_video.release()
    outputVideo.release()
    cv2.destroyAllWindows()
    
def getCoordinates(track_box):
    processVideo("video.mp4", "output.mp4")