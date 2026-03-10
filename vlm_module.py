import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
MOONDREAM_API_URL = "https://api.moondream.ai/v1/query"

print("API KEY LOADED:", MOONDREAM_API_KEY)

def analyze_scene(image_path, detections=None):

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    detected_labels = [label for label, conf in detections] if detections else []
    has_accident = "Accident" in detected_labels
    has_fire = "Car Fire" in detected_labels

    if has_fire and has_accident:
        context = "A vehicle fire and collision may be present in this image."
    elif has_fire:
        context = "A vehicle fire may be present in this image."
    elif has_accident:
        context = "A road accident may be present in this image."
    else:
        context = "This is a road or traffic scene."

    prompt = f"""
{context}

Look carefully at the entire image and describe everything you observe:

1. How many vehicles are visible? What type are they (car, truck, bike, auto)? What color are they?
2. What is the condition of the vehicles? Any damage, overturning, or collision visible?
3. Is there any fire, smoke, or flames? Where exactly and how intense?
4. What are the road conditions? Is it wet, dry, or damaged? Any signs of rain?
5. Is the road blocked or partially passable?
6. Are there any people visible? Do any appear injured or trapped?
7. What is the overall severity of the scene — minor, moderate, or severe?

Be as specific and detailed as possible. Describe only what you actually see in the image.
"""

    payload = {
        "image_url": f"data:image/jpeg;base64,{image_data}",
        "question": prompt,
        "stream": False
    }

    headers = {
        "X-Moondream-Auth": MOONDREAM_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(MOONDREAM_API_URL, json=payload, headers=headers)
    response.raise_for_status()

    return response.json().get("answer", "No response from Moondream API")