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
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

openai.api_key = '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'  # Replace with your actual API key



def plot_request_method_distribution(data):
    methods = [entry['Value']['request']['method'] for entry in data]
    print("Methods:", methods)  # Debugging line
    method_counts = pd.Series(methods).value_counts()
    print("Method Counts:", method_counts)  # Debugging line
    method_counts.plot(kind='bar')
    plt.title('Request Method Distribution')
    plt.ylabel('Number of Requests')
    plt.xlabel('HTTP Methods')
    plt.show()

def plot_response_status_distribution(data):
    statuses = [entry['Value']['response']['status'] for entry in data]
    status_counts = pd.Series(statuses).value_counts()
    status_counts.plot(kind='bar')
    plt.title('Response Status Code Distribution')
    plt.ylabel('Number of Responses')
    plt.xlabel('HTTP Status Codes')
    plt.show()

def plot_response_content_size(data):
    sizes = [entry['Value']['response']['content']['size'] for entry in data if 'size' in entry['Value']['response']['content']]
    sizes_series = pd.Series(sizes)
    sizes_series.plot(kind='hist', bins=50, logy=True)
    plt.title('Response Content Size Distribution')
    plt.ylabel('Number of Responses (Log Scale)')
    plt.xlabel('Size of Response Content (bytes)')
    plt.show()

def plot_request_timeseries(data):
    datetimes = [entry['Value']['startedDateTime'] for entry in data]
    timestamps = pd.Series(datetimes).value_counts().sort_index()
    timestamps.plot()
    plt.title('Requests Over Time')
    plt.ylabel('Number of Requests')
    plt.xlabel('Time')
    plt.show()


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
        local_vars = {"log_data": log_data}
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

def analyze_logs_and_generate_graphs(df):
    # Plotting functions
    def plot_http_methods_distribution(df, file_path):
        method_counts = df['Value.request.method'].value_counts()
        plt.figure(figsize=(10, 6))
        method_counts.plot(kind='bar', color='skyblue')
        plt.title('HTTP Methods Distribution')
        plt.xlabel('HTTP Method')
        plt.ylabel('Count')
        plt.savefig(file_path + '_methods_distribution.png')
        plt.close()

    def plot_status_code_distribution(df, file_path):
        status_counts = df['Value.response.status'].value_counts()
        plt.figure(figsize=(10, 6))
        status_counts.plot(kind='bar', color='lightcoral')
        plt.title('Response Status Codes Distribution')
        plt.xlabel('Status Code')
        plt.ylabel('Count')
        plt.savefig(file_path + '_status_code_distribution.png')
        plt.close()

    def plot_request_timeline(df, file_path):
        df['Value.startedDateTime'] = pd.to_datetime(df['Value.startedDateTime'])
        df.set_index('Value.startedDateTime', inplace=True)
        time_series = df['order'].resample('T').count()  # Resampling by minute ('T')
        plt.figure(figsize=(12, 6))
        time_series.plot()
        plt.title('Requests Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Requests')
        plt.savefig(file_path + '_request_timeline.png')
        plt.close()

    # Generate and save the plots
    plot_http_methods_distribution(df, 'http_methods_distribution')
    plot_status_code_distribution(df, 'status_code_distribution')
    plot_request_timeline(df, 'request_timeline')

    return ["http_methods_distribution.png", "status_code_distribution.png", "request_timeline.png"]

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

    # # Combine NLP processed logs with the user prompt
    # full_prompt = f"{processed_logs}\n\n{prompt_text}"
    # messages = [
    #     {"role": "system", "content": "You are a skilled backend architect and data engineer."},
    #     {"role": "user", "content": full_prompt}
    # ]
    
    # image_prompt = "Graphical representation of " + full_prompt  # Modify as needed
    # print("<><>?><><><?>",image_prompt)
    # image_response =interpret_prompt(image_prompt)
    # response = openai.chat.completions.create(
    #     model="gpt-4-1106-preview",  # Replace with the correct GPT-4 model identifier
    #     messages=messages,
    #     max_tokens=2000
    # )

    # if response.choices:
    #     response_text = response.choices[0].message.content
    # else:
    #     response_text = "No response"
    
    analyze_logs_and_generate_graphs(pd.json_normalize(log_data))
    return jsonify({"response": "response_text","image":"image_response"})

@app.route('/errorInsights', methods=['POST'])
def error_insights():
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
    analyze_logs_and_generate_graphs(pd.json_normalize(log_data))
    return jsonify({"response": "response_text","image":"image_response"})


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


