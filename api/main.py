from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os

app = FastAPI()

coordinates_storage = []

class Coordinates(BaseModel):
    x: float
    y: float
    z: float


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    with open(f"uploaded_{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_location = f"uploaded_{filename}"
    if os.path.exists(file_location):
        return FileResponse(file_location) 

    return {"error": "File not found"}
@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    video_location = f"uploaded_video_{file.filename}"
    with open(video_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.get("/download_video/{filename}")
async def download_video(filename: str):
    video_location = f"uploaded_video_{filename}"
    if os.path.exists(video_location):
        return FileResponse(video_location)
    return {"error": "File not found"}

@app.post("/upload_coordinates/")
async def upload_coordinates(coords: Coordinates):
    coordinates_storage.append(coords)
    return {"message": "Coordinates stored successfully"}

@app.get("/get_coordinates/")
async def get_coordinates():
    return coordinates_storage