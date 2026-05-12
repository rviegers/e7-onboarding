import requests

def submit_prediction(file_path, participant_name):
    url = "https://gossip-gruffly-unsocial.ngrok-free.dev/submit"
    files = {'file': open(file_path, 'rb')}
    data = {'name': participant_name}
    
    response = requests.post(url, files=files, data=data)
    return response.json()

# Example usage:
# submit_prediction('my_preds.csv', 'Team_Alpha')

if __name__ == "__main__":
    # Example usage:
    result = submit_prediction('my_preds.csv', 'Lessgooo')
    print(f"Submission score: {result['score']:.4f} | {result['message']}")
