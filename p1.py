import os
import pandas as pd
from ultralytics import YOLO
import cv2

model = YOLO("yolov8m-pose.pt")
path = "https://www.youtube.com/watch?v=RCCz96w3Lbw&ab_channel=SGYap"
results = model(path, show = True, conf = 0.3, save = True)