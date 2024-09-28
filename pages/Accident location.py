import streamlit as st
import requests
import json
import urllib.parse
from pymongo.server_api import ServerApi
import pymongo
from pymongo import MongoClient
import pandas as pd

# Load MongoDB credentials from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

username = urllib.parse.quote_plus(config["username"])
password = urllib.parse.quote_plus(config["password"])

uri = f"mongodb+srv://{username}:{password}@cluster0.ny8favk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["video_database"]
collection = db["locations"]

st.set_page_config(
    page_title="EmergencyAct",
    page_icon="🚨",
)

st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Accident Location 🚨]')

st.write("Observe most common accident locations in your area")

st.divider()

st.subheader(":orange[Pinpoint accident locations]")

# Input box for location search with suggestions
location_query = st.text_input("Search for a location")

if location_query:
    # Use OpenStreetMap API to get location suggestions
    osm_url = f"https://nominatim.openstreetmap.org/search?q={location_query}&format=json"
    response = requests.get(osm_url)
    if response.status_code == 200 and response.json():
        suggestions = response.json()
        suggestion_names = [suggestion["display_name"] for suggestion in suggestions]
        selected_suggestion = st.selectbox("Select a location", suggestion_names)

        if selected_suggestion:
            selected_location_data = next(suggestion for suggestion in suggestions if suggestion["display_name"] == selected_suggestion)
            latitude = float(selected_location_data["lat"])
            longitude = float(selected_location_data["lon"])
            st.success(f"Location selected: {selected_suggestion}")

            # Upload location to MongoDB
            location_document = {
                "name": selected_suggestion,
                "latitude": latitude,
                "longitude": longitude
            }
            collection.insert_one(location_document)
            st.success("Location uploaded to MongoDB")

            # Display location on map
            map_data = pd.DataFrame({
                'lat': [latitude],
                'lon': [longitude]
            })
            st.map(map_data)
    else:
        st.error("No suggestions found. Please try again.")

st.divider()

st.subheader(":orange[Select a stored location]")

# Fetch locations from MongoDB
locations = list(collection.find({}, {"_id": 0, "name": 1, "latitude": 1, "longitude": 1}))
location_names = [location["name"] for location in locations]

selected_location = st.selectbox("Select a location", location_names)

if selected_location:
    selected_location_data = next(loc for loc in locations if loc["name"] == selected_location)
    st.write(f"You selected: {selected_location}")
    st.map(pd.DataFrame({
        'lat': [selected_location_data["latitude"]],
        'lon': [selected_location_data["longitude"]]
    }))