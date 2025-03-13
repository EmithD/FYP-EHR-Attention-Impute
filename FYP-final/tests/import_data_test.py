import requests

url = "http://127.0.0.1:8000/api/v1/data/import-data"
files = {"file": open("physionet_wo_missing.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())