import streamlit as st
import requests
import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get analysis messages
def get_analysis_messages(image_paths, api_key):
    # Encode each image and create the messages list
    encoded_images = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(image_path)}"
            }
        } for image_path in image_paths
    ]

    # Define the prompts
    prompts = [
        "Determine possible falling reasons regarding the environment.",
        "Identify objects that were in the way.",
        "Provide recommendations on how to improve the area to prevent falls."
    ]

    # Create the messages list
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ] + encoded_images
        } for prompt in prompts
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()

    # Extract and return the output messages
    output_messages = [choice['message']['content'] for choice in response_data.get('choices', [])]

    # Create a dictionary with prompts as keys and output messages as values
    result = {prompts[i]: output_messages[i] for i in range(len(prompts))}

    return result

