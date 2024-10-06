from http.client import responses

import requests

url = "http://127.0.0.1:8001/detect"
message = "Test message;"
file_path = "sample.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, data={"message" : message}, files={"file" : file})

print(response.json())

