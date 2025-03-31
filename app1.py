import os
import json
import numpy as np
import sounddevice as sd
import whisper
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as ai
from scipy.io import wavfile
from flask import Flask, request, jsonify
import tempfile

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize Google Generative AI client
api_key = os.getenv("GOOGLE_API_KEY")
ai.configure(api_key=api_key)
model = ai.GenerativeModel("gemini-1.5-flash")

# Initialize transformers pipelines
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
intent_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

# Initialize Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# Persistent memory file path
MEMORY_FILE = "chat_history.json"

# Function to load conversation history from JSON file
def load_conversation_history():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)
    return []

# Function to save conversation history to JSON file
def save_conversation_history(messages):
    with open(MEMORY_FILE, "w") as file:
        json.dump(messages, file)

# Function to get sentiment and intent from the message
def analyze_message(message):
    sentiment = sentiment_analyzer(message)[0]['label']
    intent = intent_classifier(message)[0]['label']
    return sentiment, intent

# Function to truncate long conversation history
def truncate_history(messages, max_length=10):
    if len(messages) > max_length:
        return messages[-max_length:]
    return messages

# Function to get responses from Google's Gemini
def get_gemini_response(question, conversation_history):
    try:
        response = model.generate_content(conversation_history)
        if response:
            return response.text
        else:
            return "Error: No response from Gemini."
    except Exception as e:
        return f"Error: {str(e)}"

# Function to record audio (for demonstration, use an uploaded file instead in practice)
def record_audio(duration):
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording.flatten()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    conversation_history = load_conversation_history()

    # Truncate the history for long conversations
    conversation_history = truncate_history(conversation_history)

    # Analyze the message for sentiment and intent
    sentiment, intent = analyze_message(user_input)

    # Get response from Google Gemini
    response = get_gemini_response(user_input, conversation_history)

    # Append user message and assistant response to conversation history
    conversation_history.append({"role": "user", "content": user_input, "sentiment": sentiment, "intent": intent})
    conversation_history.append({"role": "assistant", "content": response})

    # Save conversation history to the persistent memory
    save_conversation_history(conversation_history)

    return jsonify({'response': response})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio_file.save(temp_file.name)
    audio_input = whisper_model.transcribe(temp_file.name)
    os.remove(temp_file.name)  # Clean up the temporary file
    return jsonify({'transcribed_text': audio_input['text']})

if __name__ == '__main__':
    app.run(debug=True)
