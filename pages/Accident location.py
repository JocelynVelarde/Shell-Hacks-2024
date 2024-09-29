import streamlit as st
import json
import urllib.parse
from algorithms.BB_prompt import AccidentPrompt
from pymongo.server_api import ServerApi
from pymongo import MongoClient
import pandas as pd

api_key = st.secrets["OPEN_AI_KEY"]

with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

uri = f"mongodb+srv://{username}:{password}@cluster0.6veno.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["video_database"]
collection = db["locations_camera"]

st.set_page_config(
    page_title="EmergencyAct",
    page_icon="🚨",
)
st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Accident Location 🚨]')

st.write("Observe most common accident locations in your area")

st.divider()

st.subheader(":orange[Add a new camera location]")

latitude = st.number_input("Enter latitude", format="%.6f")
longitude = st.number_input("Enter longitude", format="%.6f")
camera = st.text_input("Enter camera name")

if st.button("Add Camera"):
    if latitude and longitude and camera:
        st.success(f"Camera pinned: Latitude {latitude}, Longitude {longitude}")

        location_document = {
            "latitude": latitude,
            "longitude": longitude,
            "camera": camera
        }
        collection.insert_one(location_document)
        st.success("Camera location uploaded successfully!")

        map_data = pd.DataFrame({
            'lat': [latitude],
            'lon': [longitude]
        })
        st.write("Map data:", map_data)
        st.map(map_data)
else:
    st.error("Please enter both latitude and longitude.")

st.divider()

st.subheader(":orange[Select a stored camera location]")

locations = list(collection.find({}, {"_id": 0, "latitude": 1, "longitude": 1, "camera": 1}))
location_names = [f"Camera: {loc['camera']}, Lat: {loc['latitude']}, Lon: {loc['longitude']}" for loc in locations]

selected_location = st.selectbox("Select a location", location_names)

if selected_location:
    selected_location_data = next(loc for loc in locations if f"Camera: {loc['camera']}, Lat: {loc['latitude']}, Lon: {loc['longitude']}" == selected_location)
    st.write(f"You selected: {selected_location}")
    map_data = pd.DataFrame({
        'lat': [selected_location_data["latitude"]],
        'lon': [selected_location_data["longitude"]],
        'camera': [selected_location_data["camera"]]
    })
    st.write("Map data:", map_data)
    st.map(map_data)


st.divider()

st.subheader(":orange[Locate where accidents happen]")

st.write("Use a get to obtain the bounding box corners of the accident location from mongo")

st.divider()

st.subheader(":orange[Risk Analysis using camera location and person position]")

lat = 25
lon = 25
x1 = 10
y1 = 10
x2 = 20
y2 = 20


if st.button("Find relation between camera and person"):
    if lat and lon and x1 and y1 and x2 and y2 and api_key:
        result = AccidentPrompt(lat, lon, x1, y1, x2, y2, api_key)
        st.write("Answer:")
        st.write(result)
    else:
        st.error("Please enter all required fields.")


