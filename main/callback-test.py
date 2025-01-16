import requests
import json

def send_diagnosis_callback(session_id, callback_url, diagnosis_results):
    """
    Sends diagnosis results to the callback API.

    Args:
        session_id (str): The session ID, used to build the user_seq in the callback URL.
        callback_url (str): The base callback URL.
        diagnosis_results (list): A list of diagnosis results containing task, code, and probability.

    Returns:
        dict: The response from the API.
    """
    # Extract user_seq from session_id 
    full_url = f"{callback_url}{session_id}"
    print(full_url)

    # Prepare the headers and payload
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    payload = json.dumps(diagnosis_results)

    # Make the PUT request
    try:
        response = requests.put(full_url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

# Example usage
data = {
    "session_id": "20241217011223",
    "callback": "https://dev.mentai.waymed.ai:18040/api/v1/diagnosis/",
}

diagnosis_results = [
    {
        "task": "Task1",
        "prediction": "SAD",
        "probability": 0.89
    },
    {
        "task": "Task3",
        "prediction": "NORMAL",
        "probability": 0.81
    }
]

response = send_diagnosis_callback(data["session_id"], data["callback"], diagnosis_results)
print(response)