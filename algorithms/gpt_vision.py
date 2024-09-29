import streamlit as st
import requests
import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get analysis messages
def get_analysis_messages(image_paths, api_key):
    # Encode each image
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

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    result = {}

    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ] + encoded_images
            }
        ]

        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()

        # Extract the output message
        output_message = response_data['choices'][0]['message']['content']
        result[prompt] = output_message

    return result

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
    respuesta = get_analysis_messages([image_paths[0]], api_key)[0]
    st.subheader("Answer:")
    st.write(respuesta)