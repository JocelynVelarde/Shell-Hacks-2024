import streamlit as st
import pymongo
import gridfs
import urllib.parse
import json
from moviepy.editor import VideoFileClip
from algorithms.inference_static import download_video_from_mongoDB, process_video, upload_video_to_mongoDB

# Load MongoDB credentials from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(uri)
db = client["video_database"]
fs = gridfs.GridFS(db)

st.set_page_config(
    page_title="EmergencyAct",
    page_icon="🚨",
)

st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Video analysis 🚨]')

st.write("You can see the different metrics collected from cameras here")

st.divider()

st.subheader(":orange[Visualize your raw video]")

uploaded_file = st.file_uploader("Upload a video", type=["mp4, avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Upload Video"):
        video_location = f"uploaded_video_{uploaded_file.name}"
        file_id = fs.put(uploaded_file.getvalue(), filename=uploaded_file.name)
        
        video_metadata = {
            "filename": uploaded_file.name,
            "file_id": file_id
        }
        db.videos.insert_one(video_metadata)
        st.success("Video uploaded successfully!")

st.divider()

st.subheader(":orange[Select an uploaded video]")

# Fetch the list of uploaded videos
video_list = [video["filename"] for video in db.videos.find()]
selected_video = st.selectbox("Select a video", video_list)
st.write(f"You selected: {selected_video}")

if st.button("Download and Analyze Video"):
    video_metadata = db.videos.find_one({"filename": selected_video})
    if video_metadata:
        st.write(selected_video)
        file_id = video_metadata["file_id"]
        grid_out = fs.get(file_id)
        video_bytes = grid_out.read()
        with open(selected_video, "wb") as f:
            f.write(video_bytes)

        # Convert the .avi video to .mp4 format
        clip = VideoFileClip(selected_video)
        mp4_video = selected_video.replace(".avi", ".mp4")
        clip.write_videofile(mp4_video, codec="libx264")
        
        # Process the video
        download_video_from_mongoDB(uri, selected_video)
        output_video = process_video(selected_video)
        upload_video_to_mongoDB(uri, output_video)
        
        st.success("Video downloaded successfully and uploaded to DB!")
        
        # Display the original video
        with open(mp4_video, "rb") as f:
            mp4_video_bytes = f.read()
        st.video(mp4_video_bytes)
        
        # Read and display the processed video
        with open(output_video, "rb") as f:
            output_video_bytes = f.read()
        st.video(output_video_bytes)
    else:
        st.error("Failed to download video.")
