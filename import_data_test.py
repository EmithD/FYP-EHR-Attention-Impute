import requests

url = "http://127.0.0.1:8000/upload"
files = {"file": open("G:/FYP/FYP-EHR-Attention-Impute/FYP-final/tests/physionet_wo_missing.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())