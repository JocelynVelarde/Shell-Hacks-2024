import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
        page_title="EmergencyAct",
        page_icon="🚨",
)

st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Welcome to EmergencyAct 🚨]')

st.write('We use your surveillance cameras to detect accidents and emergencies in real time. We are here to help you!')

st.subheader('We have two main infrastructure features:')

col1, col2 = st.columns(2)
col1.subheader(":orange[Computer Vision Tools]")
col1.divider()
col1.write("- Use of object detection to detect variety of people in an area")
col1.write("- Obtain pose estimation to detect if a person is in a dangerous position")
col1.write("- Mantain privacy of people using only tags and not saving biometric data")
col1.write("- Prompt pose coordinate change to determine position of the person using LLM")
col1.write("- Reccomendations on what to change to prevent accidents")

col2.subheader(":orange[Generative AI Tools]")
col2.divider()
col2.write("- Use vision AI for snapshots to detect timestamp of fall")
col2.write("- Obtain relative position of the accident location and the person")
col2.write("- Use timestamp to analyze possible causes of the accident")
col2.write("- Prompt crafting using above data to call emergency services using text to speech")
col2.write("- Dashboard to show summary of accidents and emergencies")

st.divider()
st.subheader(":orange[We combine the use of several tools to make your spaces safer]")

st.write("📌 Computer Vision: OpenCV")
st.write("📌 Natural Language Processing: GPT-3.5 Turbo, prompt classification")
st.write("📌 Generative AI: GPT-4-o, base64 image encoding")
st.write("📌 Data Visualization: Plot")
st.write("📌 Model Vision YOLOv8mpose, ultralytics, pytorch")
st.write("📌 Machine Learning Random Forest (97% test set accuracy), MLP (97% test set accuracy)")
st.write("📌 libraries streamlit, CV2, ultralytics, numpy, base64, os, requests, openai, collections, datetime, torch, pickle, sklearn, joblib, pandas, numpy, fastapi, shutil, pymongo, urllib, aiohttp, json, PIL, gridfs, BASE64.")

st.divider()
st.subheader(":orange[Get Started]")
st.page_link("pages/instructions.py", label="See Instructions 🚀")

st.divider()

st.markdown("<span style='margin-left: 250px; font-weight: 20px; font-size: 15px'>Thanks for using EmergenceyAct 🚨</span>", unsafe_allow_html=True)
