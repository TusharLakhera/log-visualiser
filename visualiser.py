from flask import Flask, request, jsonify
import openai
import json
import spacy
from base64 import b64encode
import boto3
import io
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

openai.api_key = 'API-KEY'  # Replace with your actual API key

def generate_image(prompt):
    try:
        response = openai.images.create(
            model="dall-e-2021-12-01",  # Use the correct DALL-E model
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response
    except Exception as e:
        print("Error in generating image:", e)
        return None

def preprocess_logs(logs):
    processed_logs = []

    for entry in logs:
        # Extract relevant information
        request = entry.get('Value', {}).get('request', {})
        response = entry.get('Value', {}).get('response', {})

        url = request.get('url', 'No URL')
        method = request.get('method', 'No Method')
        status = response.get('status', 'No Status')
        status_text = response.get('statusText', 'No Status Text')

        # Create a summarized version of the log entry
        summary = f"URL: {url}, Method: {method}, Status: {status} {status_text}"
        processed_logs.append(summary)

    return ' '.join(processed_logs)

@app.route('/analyze', methods=['POST'])
def analyze_logs():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    test_id = data.get('testId')
    prompt_text = data.get('prompt')

    if not test_id:
        return jsonify({"error": "No testId provided"}), 400
    if not prompt_text:
        return jsonify({"error": "No prompt provided"}), 400
    session = boto3.Session()
    s3 = session.client('s3')

    bucket_name = 'hackathon-logs18'  # Replace with your bucket name
    object_key =  'GHIJ-LJKH-HUIO/network.json'  # Adjust the path as needed

    # Download the file
    file_content = s3.get_object(Bucket=bucket_name, Key=object_key)['Body'].read()

    # Assuming the file content is a JSON string
    log_data = json.loads(file_content)
    print(log_data)
    processed_logs=preprocess_logs(log_data)

    # Combine NLP processed logs with the user prompt
    full_prompt = f"{processed_logs}\n\n{prompt_text}"
    print(full_prompt)
    # Prepare messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a skilled bakckend architect and data engineer."},
        {"role": "user", "content": full_prompt}
    ]
    
    image_prompt = "Graphical representation of " + full_prompt  # Modify as needed

    # Generate an image using DALL-E
    # image_response = generate_image(image_prompt)

    # # Send messages to GPT-4 and get response
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",  # Replace with the correct GPT-4 model identifier
        messages=messages,
        max_tokens=2000
    )
    # response = openai.images.generate(
    #     model="gpt-4-1106-preview",
    #     prompt=response,
    #     size="1024x1024",
    #     quality="standard",
    #     n=1,)
    if response.choices:
        response_text = response.choices[0].message.content
    else:
        response_text = "No response"

    return jsonify({"response": response_text,
    "image":image_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

