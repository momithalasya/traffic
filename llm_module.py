import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")


def generate_response(scene_report, detections):

    # Format detections clearly with confidence %
    detection_lines = "\n".join(
        f"  - {label} (confidence: {conf*100:.1f}%)"
        for label, conf in detections
    )

    has_fire = any(label == "Car Fire" for label, _ in detections)
    has_accident = any(label == "Accident" for label, _ in detections)

    # Set incident type string for prompt clarity
    if has_fire and has_accident:
        incident_type = "ACCIDENT WITH FIRE"
    elif has_fire:
        incident_type = "CAR FIRE"
    else:
        incident_type = "ROAD ACCIDENT"

    prompt = f"""
You are an AI traffic emergency dispatch system. You receive inputs from a YOLO detector and a visual scene analyzer.
Your job is to assess the situation and dispatch the right response FAST.

=== INCIDENT TYPE ===
{incident_type}

=== YOLO DETECTIONS ===
{detection_lines}

=== VISUAL SCENE ANALYSIS ===
{scene_report}

=== YOUR TASK ===
Based on the above, respond ONLY in this exact format:

Emergency Level: [Low / Medium / High / Critical]
Reason: [One line explaining why this level]

Units To Dispatch:
- [Unit name] — [specific reason based on scene]
- [Unit name] — [specific reason based on scene]

Immediate Actions (next 5 minutes):
1.
2.
3.

Traffic Control:
- [Specific road/lane action]
- [Specific road/lane action]

Estimated Scene Clear Time: [X minutes]

Be specific, operational, and concise. No generic advice.
"""

    response = model.generate_content(prompt)
    return response.text