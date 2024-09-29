import streamlit as st
from algorithms.gpt_vision import get_analysis_messages
from sentence_transformers import SentenceTransformer
import requests
import faiss
import numpy as np
from PIL import Image
import io

api_key = st.secrets["OPEN_AI_KEY"]

# Initialize the vector database
model = SentenceTransformer('clip-ViT-B-32')
index = faiss.IndexFlatL2(512)  # Assuming 512-dimensional embeddings

# Streamlit app
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
        image = Image.open(uploaded_file)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()
        
        # Encode the image
        image_embedding = model.encode([image_bytes])
        index.add(image_embedding)
        
        image_paths.append(uploaded_file.name)

    st.write("Images uploaded and encoded successfully.")

query = st.text_input("Ask a question about the images")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 relevant images

    retrieved_images = [image_paths[i] for i in I[0]]
    st.write("Retrieved Images:")
    for img_path in retrieved_images:
        st.image(img_path)

    # Generate responses using the retrieved images
    encoded_images = [{"type": "image", "image": img_path} for img_path in retrieved_images]
    prompts = [
        "Determine possible falling reasons regarding the environment.",
        "Identify objects that were in the way.",
        "Provide recommendations on how to improve the area to prevent falls."
    ]

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

    for i, message in enumerate(response_data['choices'][0]['message']['content'], 1):
        st.subheader(f"Message {i}")
        st.write(message)