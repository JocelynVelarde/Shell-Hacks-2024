import streamlit as st
import requests

st.set_page_config(
    page_title="EmergencyAct",
    page_icon="🚨",
)

st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Video analysis 🚨]')

st.write("You can see the different metrics collected from cameras here")

st.divider()

st.subheader(":orange[Visualize your raw video]")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Upload Video"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
        response = requests.post("http://0.0.0.0:8000/upload_video/", files=files)
        if response.status_code == 200:
            st.success("Video uploaded successfully!")
            video_id = response.json().get("video_id")
            st.session_state["video_id"] = video_id
        else:
            st.error("Failed to upload video.")

st.divider()

st.subheader(":orange[Select an uploaded video]")

# Fetch the list of uploaded videos
list_videos_response = requests.get("http://0.0.0.0:8000/list_videos/")
if list_videos_response.status_code == 200:
    video_list = list_videos_response.json().get("videos", [])
    selected_video = st.selectbox("Select a video", video_list)
    st.write(f"You selected: {selected_video}")

    if st.button("Analyze Selected Video"):
        analysis_response = requests.post(f"http://0.0.0.0:8000/analyze_video/{selected_video}")
        if analysis_response.status_code == 200:
            st.success("Video analyzed successfully!")
            st.write(analysis_response.json())
        else:
            st.error("Failed to analyze video.")
else:
    st.error("Failed to fetch the list of videos.")