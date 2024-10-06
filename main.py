from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

app = FastAPI()

model = YOLO('yolov8n.pt')

class DetectionModel(BaseModel):
    message: str
    image: str

def detect_objects(image: Image.Image):
    img = np.array(image)
    results = model(img)
    class_names = model.names

    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = class_names[int(class_id)]
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}',
                        (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            result_image = Image.fromarray(img)
            return result_image


@app.get('/')
async def read_root():
    return {"Hello" : "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id" : item_id, "q" : q}

@app.post("/detect", response_model=DetectionModel)
async def detect_service(message: str = Form(...), file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    result_image = detect_objects(image)

    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    #.decode('utf-8')에 관하여...base64 인코딩 후 인코딩된 데이터가 바이트 객체를 반환하기 때문에
    # 바이트 객체를 문자열 형태로 사용하려면 디코딩이 필요하기 때문임.
    return DetectionModel(message=message, image=img_str)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)

