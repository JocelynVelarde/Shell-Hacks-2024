from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import urllib.parse
import json
import aiohttp

app = FastAPI()


# Load MongoDB credentials from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["video_database"]
collection = db["videos"]

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    video_location = f"uploaded_video_{file.filename}"
    with open(video_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_metadata = {
        "filename": file.filename,
        "location": video_location
    }
    collection.insert_one(video_metadata)
    
    return {"filename": file.filename}

@app.get("/list_videos/")
async def list_videos():
    videos = []
    for video_metadata in collection.find():
        videos.append(video_metadata["filename"])
    return {"videos": videos}

@app.post("/check_video_exists/{video_id}")
async def analyze_video(video_id: str):
    video_metadata = collection.find_one({"filename": video_id})
    if video_metadata:
        return {"message": "Video found in MongoDB"}
    return {"error": "Video not found in MongoDB"}

async def get_video_path(request: Request):
    return await request.json()

@app.post("/analyze_video")
async def analyze_video(request: Request):
    # Get the video path from the request data
    video_path = await get_video_path(request)

    # Send an asynchronous HTTP request to another machine (e.g., using SSH or another API)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://172.18.0.1/analyze_video",
            json={"video_path": video_path},
            headers={"Authorization": "Bearer YOUR_API_KEY"}
        ) as response:
            return {"result": await response.json()}


@app.get("/download_video/{filename}")
async def download_video(filename: str):
    video_metadata = collection.find_one({"filename": filename})
    if video_metadata:
        video_location = video_metadata["location"]
        if os.path.exists(video_location):
            return FileResponse(video_location)
    return {"error": "File not found"}

@app.post("/upload_coordinates/")
async def upload_coordinates():
    return {"message": "Coordinates uploaded successfully"}