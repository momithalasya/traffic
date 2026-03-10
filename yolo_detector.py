from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

class_names = {
    0: "Accident",
    1: "Car Fire"
}

def detect_objects(image_path):

    results = model(image_path)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            label = class_names.get(cls, "Unknown")

            detections.append((label, conf))

    return detections