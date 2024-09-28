from openai import OpenAI

client = OpenAI(api_key="")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are in these images? Is there any difference between them?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "assets/images/cds100001.png",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "assets/images/cds100004.png",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "assets/images/cds100035.png",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "assets/images/cds100047 1.png",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "assets/images/cds100077.png",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0])