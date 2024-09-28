from ultralytics import YOLO

model = YOLO("models/yolov8-pose.pt")

video_url = 'https://www.youtube.com/watch?v=nYg6L-CPqww&ab_channel=NBCNews'

results = model(video_url, save=True, show=True, conf=0.5)