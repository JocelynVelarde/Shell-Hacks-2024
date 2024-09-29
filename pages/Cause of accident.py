import streamlit as st
import json
import urllib.parse
from algorithms.gpt_vision import get_analysis_messages, chat_prompt
import pymongo
import gridfs

# Load MongoDB credentials from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(uri)
db = client["video_database"]
fs = gridfs.GridFS(db)

json_data = []
def download_json_from_mongoDB():
    
    for file in fs.find():
        if file.filename.endswith(".json"):
            json_data.append(file.read())
    return json_data

# Example usage in Streamlit app
api_key = st.secrets["OPEN_AI_KEY"]

st.set_page_config(
    page_title="EmergencyAct",
    page_icon="🚨",
)
st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Cause of Accident 🚨]')

st.write("Observe the photo snapshots of the accident and analyze the possible causes to obtain recommendations on what to improve")

st.divider()

uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    image_paths = []
    for uploaded_file in uploaded_files:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
            image_paths.append(f"temp_{uploaded_file.name}")

    analysis_results = get_analysis_messages(image_paths, api_key)
    
    for i, (prompt, message) in enumerate(analysis_results.items(), 1):
        st.subheader(f"Prompt {i}: {prompt}")
        st.write(message)

st.divider()
st.subheader(':orange[Ask questions about the accident]')
chat_input = st.text_area("Type your question here")
if st.button("Send"):
        chat_response = chat_prompt(analysis_results, chat_input, api_key)
        st.subheader("Answer:")
        st.write(chat_response)
