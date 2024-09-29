import openai

def bounding_box_prompt(lat, lon, x1, y1, x2, y2, api_key):
    openai.api_key = api_key
    
    box_prompt = """
    Given the latitude and longitude of a camera, along with the bounding box coordinates (x1, y1) and (x2, y2) representing the top-left and bottom-right corners of a person's bounding box in an image captured by the camera, estimate the person's real-world position (latitude and longitude). Leverage the camera’s field of view, image resolution, and optionally, the person’s height or depth data to convert the bounding box coordinates into real-world angles and distance. Use spherical geometry to calculate the person’s geographic position relative to the camera.

    Provide a detailed, step-by-step explanation of your approach, including:

    How to compute the center of the bounding box and convert it to angles (azimuth and elevation) using the camera's field of view and image resolution.
    How to estimate the distance to the person, factoring in optional height or depth information.
    How to convert the calculated angles and distance into the person's real-world latitude and longitude using spherical geometry.
    Any additional considerations, such as the camera's elevation or tilt.

    Latitude: {}
    Longitude: {}
    Bounding Box Coordinates:
    Top-left: ({}, {})
    Bottom-right: ({}, {})
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": box_prompt + lat + lon + x1 + y1 + x2 + y2}],
        max_tokens=600,
        temperature=0.5
    )
    
    print("Position of the person: ", response.choices[0].message['content'])
    message = response.choices[0].message['content']
    return message