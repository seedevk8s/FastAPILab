import requests
from torchvision import message

url = "http://127.0.0.1:8080/detect"
message = "qweretyuikl;"
file_path = "sample.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, data={})