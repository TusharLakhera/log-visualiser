from flask import Flask, request, jsonify
import openai
import json
import spacy
from base64 import b64encode
import boto3
import io,os
import re
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from collections import Counter
import base64
import pandas
from urllib.parse import urlparse
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

openai.api_key = 'API_KEY'  # Replace with your actual API key



def interpret_prompt(prompt):
    try:
        # Formulate the prompt for OpenAI
        ai_prompt = f"Act Like a data engineer and backend architect and dont give any text just the python function which i can use directly to plot a graph: '{prompt}'?"
        messages = [
        {"role": "system", "content": "You are a skilled backend architect and data engineer."},
        {"role": "user", "content": ai_prompt}]

        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=2000)
        response_code = response.choices[0].message.content
        cleaned_code = response_code.replace('```python', '').replace('```', '').strip()
        print(cleaned_code)
        # Execute the cleaned Python code
        # Define a local dictionary where the executed code can write its output
        local_vars = {}
        exec(cleaned_code, globals(), local_vars)
        # Assuming the code generates a matplotlib figure
        # Convert the figure to an image (PNG) in base64 encoding
        image_path = '/Users/tusharl/Desktop/projects/graph.png'  # Change the path as needed
        plt.savefig(image_path)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return f"<img src='data:image/png;base64,{image_base64}' />"
    except Exception as e:
        print("Error in running the provided code:", e)
        return "Error in generating graph"

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
    processed_logs=preprocess_logs(log_data)

    # Combine NLP processed logs with the user prompt
    full_prompt = f"{processed_logs}\n\n{prompt_text}"
    messages = [
        {"role": "system", "content": "You are a skilled backend architect and data engineer."},
        {"role": "user", "content": full_prompt}
    ]
    
    image_prompt = "Graphical representation of " + full_prompt  # Modify as needed
    print("<><>?><><><?>",image_prompt)
    image_response =interpret_prompt(image_prompt)
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",  # Replace with the correct GPT-4 model identifier
        messages=messages,
        max_tokens=2000
    )

    if response.choices:
        response_text = response.choices[0].message.content
    else:
        response_text = "No response"

    return jsonify({"response": response_text,"image":image_response})


@app.route('/test-credentials', methods=['GET'])
def test_credentials():
    access_key = os.environ.get('AWS_ACCESS_KEY_ID', 'Not Found')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', 'Not Found')
    return jsonify({
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)


