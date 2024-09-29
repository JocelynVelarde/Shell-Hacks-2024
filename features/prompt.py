from openai import OpenAI
import streamlit as st

def EmergencyPrompt(Nose, Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow, Right_Hip, Left_Hip, Right_Foot, Left_Foot):
    # API 
    client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

    # Body Position Detection
    body_position_prompt = """
    Given an array representing the 2D positions of key body parts for a person, where each index corresponds to a specific joint:
        Nose (0), Right Shoulder (5), Left Shoulder (6), Right Elbow (7), Left Elbow (8),
        Right Hip (11), Left Hip (12), Right Foot (15), and Left Foot (16).
    Each position is represented as a coordinate pair [x, y], such as [x0, y0] for the Nose, [x5, y5] for the Right Shoulder, 
    and [x11, y11] for the Right Hip.

    Using these coordinates, classify the person's posture into one of the following categories:
        'standing', 'sitting', 'lying down', 'falling', 'running', 'walking' or 'jumping'.

    Consider the following factors:
    - Relative distances between key joints (e.g., Shoulder-Hip ratio, Hip-Foot distance)
    - Vertical alignment of the body (e.g., Nose-Shoulder-Hip positions along the y-axis)
    - Angles between joints (e.g., Shoulder-Elbow, Hip-Knee angles)

    Provide a step-by-step explanation of your approach:
    1. Calculate the distances between relevant joints.
    2. Determine the angles formed by arms, torso, and legs.
    3. Use these metrics to identify posture patterns (e.g., close Shoulder-Hip distances indicate a sitting posture).
    4. Return a labeled classification ('standing', 'sitting', 'lying down', 'falling', 'running', 'walking' or 'jumping').
    5. Provide a summary of the decision logic.
    """
    
    # Making the request to the API
    body_response = client.ChatCompletion.create(
        model="text-davinci-003",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": body_position_prompt}
        ],
        max_tokens=600,
        temperature=0.5
    )
    
    def type_of_movement(body_response):
        if 'fell' in body_response.choices[0].message["content"]:
            return sms('fell')
        elif 'lying down' in body_response.choices[0].message["content"]:
            return sms('lying down')

    def person_in_danger(body_response):
        if 'blood detected' in body_response.choices[0].message["content"]:
            return sms('blood detected')
            
    # It prints the request
    print("Body Position Detection Output:")
    return print(body_response.choices[0].message["content"])