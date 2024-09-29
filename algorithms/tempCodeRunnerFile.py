import pymongo
import gridfs
import json
import urllib.parse
import os

with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

video_path = "test.mp4"
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