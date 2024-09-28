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

col3, col4 = st.columns(2)
col3.subheader("📌 Computer Vision")
col3.write("1. OpenCV")
col3.write("2. Example")


col4.subheader("📌 Natural Language Processing")
col4.write("1. gpt-3.5 turbo")
col4.write("2. Example")

col5, col6 = st.columns(2)
col5.subheader("📌 Generative AI")
col5.write("1. gpt4-o")
col5.write("2. Example")

col6.subheader("📌 Data Visualization")
col6.write("1. Plotly")
col6.write("2. Example")

st.divider()
st.subheader(":orange[Get Started]")
st.page_link("pages/instructions.py", label="See Instructions 🚀")

st.divider()

st.markdown("<span style='margin-left: 250px; font-weight: 20px; font-size: 15px'>Thanks for using EmergenceyAct 🚨</span>", unsafe_allow_html=True)
