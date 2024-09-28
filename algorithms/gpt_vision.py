import requests
import base64

# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# List of image paths
image_paths = [
    "assets/images/cds100001.png",
    "assets/images/cds100004.png",
    "assets/images/cds100035.png",
    "assets/images/cds100047.png",
    "assets/images/cds100048.png"
]

# Encode each image and create the messages list
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Determine why the person fell and possible causes/objects/place of the accident"
            }
        ] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(image_path)}"
                }
            } for image_path in image_paths
        ]
    }
]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4o-mini",
    "messages": messages,
    "max_tokens": 100
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json())