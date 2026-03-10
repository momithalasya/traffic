from yolo_detector import detect_objects
from vlm_module import analyze_scene
from llm_module import generate_response

image = "im1.png"

detections = detect_objects(image)
print("Detections:", detections)

incident_detected = any(conf > 0.5 for _, conf in detections)

if incident_detected:
    print("\n Incident detected")
    print("\nSending image to VLM...\n")

    # Pass detections so VLM uses a focused prompt
    scene_description = analyze_scene(image, detections=detections)

    print("Scene description:")
    print(scene_description)

    print("\nSending scene to LLM...\n")
    response = generate_response(scene_description, detections)

    print("LLM Response:")
    print(response)

else:
    print("\n✅ No incident detected")