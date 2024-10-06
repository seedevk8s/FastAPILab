import paho.mqtt.client as matt
import base64
from PIL import Image
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

from main import model

model = YOLO('yolov8n.pt')

broker = 'localhost'
port = 1883
topic = '/camera/objects'

client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

client.on_connect = on_connect
client.connect(broker, port, 60)

def get_colors(num_colors):
    np.random.seed(0)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_colors)]
    return colors

class_names = model.names
num_classes = len(class_names)
colors = get_colors(num_classes)

def detect_objects(image: np.array):
    #img = np.array(image)
    results = model(image, verbose=False)
    class_names = model.names

    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = class_names[int(class_id)]
            cv2.rectangle(image, (x1,y1), (x2,y2), colors[int(class_id)], 2)
            cv2.putText(image, f'{label} {confidence:.2f}',
                        (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[int(class_id)], 2)

    #result_image = Image.fromarray(img)
    return image

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_image = detect_objects(frame)

    _, buffer = cv2.imencode('.jpg', result_image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    payload = json.dumps({'image': jpg_as_text})
    client.publish(topic, payload)

    cv2.imshow('Frame', np.array(result_image))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()

























