from openai import OpenAI
import streamlit as st

def bounding_box_prompt(lat, lon, x1, y1, x2, y2):
    # x1, y1 are the points at the top left corner of the bounding box
    # x2, y2 are the points at the bottom right corner of the bounding box
    client = OpenAI(api_key = st.secrets["OPENAI_KEY"])
     
    box_prompt = """
    Given the latitude and longitude of a camera, along with the bounding box coordinates (x1, y1) and (x2, y2) representing the top-left and bottom-right corners of a person's bounding box in an image captured by the camera, estimate the person's real-world position (latitude and longitude). Leverage the camera’s field of view, image resolution, and optionally, the person’s height or depth data to convert the bounding box coordinates into real-world angles and distance. Use spherical geometry to calculate the person’s geographic position relative to the camera.

    Provide a detailed, step-by-step explanation of your approach, including:

    How to compute the center of the bounding box and convert it to angles (azimuth and elevation) using the camera's field of view and image resolution.
    How to estimate the distance to the person, factoring in optional height or depth information.
    How to convert the calculated angles and distance into the person's real-world latitude and longitude using spherical geometry.
    Any additional considerations, such as the camera's elevation or tilt.
    """
    
    response = client.chat.box_prompt.create(
        engine = "text-davinci-003",
        prompt = box_prompt,
        max_tokens = 600,
        temperature = 0.5
    )
    
    
    return message

    