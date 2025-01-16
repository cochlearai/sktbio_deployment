import requests

# URL for the API endpoint
# url = "http://11.0.0.7:9000/sadvoice"
url = "http://localhost:9000/sadvoice"
# url = "http://10.0.0.7:5000/sadvoice"


# Payload data
data = {
    "session_id": "1",
    "callback": "https://dev.mentai.waymed.ai:18040/api/v1/diagnosis/",
    "task1_files": {
        "1": "test202501131257/1-1.wav",
        "2": "test202501131257/1-2.wav",
        "3": "test202501131257/1-3.wav",
        "4": "test202501131257/1-4.wav",
        "5": "test202501131257/1-5.wav",
        "6": "test202501131257/1-6.wav",
        "7": "test202501131257/1-7.wav",
        "8": "test202501131257/1-8.wav",
        "9": "test202501131257/1-9.wav"
    },
    "task3_files": {
        "1": "test202501131257/3-1.wav",
        "2": "test202501131257/3-2.wav",
        "3": "test202501131257/3-3.wav",
        "4": "test202501131257/3-4.wav",
        "5": "test202501131257/3-5.wav",
        "6": "test202501131257/3-6.wav",
        "7": "test202501131257/3-7.wav"
    },
    "thresholds": {
        "THRESHOLD_TASK1":"0.5",
        "THRESHOLD_TASK3_AUDIO":"0.5",
        "THRESHOLD_TASK3_LLM":"0.5",
        "THRESHOLD_TASK3_VOTING":"0.5"
    }
}

import time
try:
    # Send the POST request
    t = time.time()
    response = requests.post(url, json=data)
    s = time.time()
    print(s-t)
    # Print the response
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
