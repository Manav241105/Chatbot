import streamlit as st
import os
import json
import numpy as np
import sounddevice as sd
import whisper  # For speech recognition
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as ai
import tempfile
from scipy.io import wavfile  # Import wavfile from scipy

# Load environment variables
load_dotenv()

# Initialize Google Generative AI client
api_key = os.getenv("GOOGLE_API_KEY")  # Ensure you have your API key in .env
ai.configure(api_key=api_key)
model = ai.GenerativeModel("gemini-1.5-flash")

# Initialize transformers pipelines
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
intent_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")  # Modify as needed

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
        return messages[-max_length:]  # Only keep the last 'max_length' messages
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

# Function to record audio
def record_audio(duration):
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    return recording.flatten()

# Function to process the user question and generate a response
def process_user_input(user_input):
    # Truncate the history for long conversations
    st.session_state['flowmessages'] = truncate_history(st.session_state['flowmessages'])

    # Build the conversation context as a string
    conversation_history = ""
    for message in st.session_state['flowmessages']:
        if message['role'] == 'user':
            conversation_history += f"User: {message['content']}\n"
        elif message['role'] == 'assistant':
            conversation_history += f"Assistant: {message['content']}\n"

    # Add current question to conversation history
    conversation_history += f"User: {user_input}\n"

    # Analyze the message for sentiment and intent
    sentiment, intent = analyze_message(user_input)

    # Get response from Google Gemini
    response = get_gemini_response(user_input, conversation_history)

    # Append question and response to session state for continuity
    st.session_state['flowmessages'].append({"role": "user", "content": user_input, "sentiment": sentiment, "intent": intent})
    st.session_state['flowmessages'].append({"role": "assistant", "content": response})

    # Save conversation history to the persistent memory
    save_conversation_history(st.session_state['flowmessages'])

    return response

# Set up Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot", layout="wide")
st.title("💬 Conversational Q&A Chatbot")

# Sidebar for options
with st.sidebar:
    st.header("Options")
    clear_chat = st.button("Clear Conversation")
    st.write("Customize your chat experience.")

# Load or initialize session state for storing chat messages
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = load_conversation_history()

# Clear conversation if button is clicked
if clear_chat:
    st.session_state['flowmessages'] = []
    save_conversation_history([])  # Clear the persistent memory

# Input box for user question
input_placeholder = "Type your question here..."
user_input = st.text_input("Ask your question: ", value="", key="input", placeholder=input_placeholder)

# Ask button
submit = st.button("Ask")

# Button to record live audio input
record_audio_button = st.button("Record Live Audio")

if record_audio_button:
    with st.spinner("Recording..."):
        # Record audio for 5 seconds (adjust as needed)
        audio_data = record_audio(5)
        st.success("Recording complete!")

        # Save audio data to a temporary file for Whisper to process
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wavfile.write(temp_file.name, 44100, (audio_data * 32767).astype(np.int16))  # Scale to int16

        # Transcribe audio using Whisper
        audio_input = whisper_model.transcribe(temp_file.name)
        st.write(f"Transcribed text: {audio_input['text']}")

        # Set the transcribed text directly as the input value
        user_input = audio_input['text']  # Store transcribed text temporarily for this iteration

        # Process the user input to get a response
        response = process_user_input(user_input)
        st.subheader("🤖 Assistant Response:")
        st.write(response)

# If submit button is clicked and user input is provided
if submit and user_input:
    with st.spinner("Generating response..."):
        response = process_user_input(user_input)
        st.subheader("🤖 Assistant Response:")
        st.write(response)

# Display conversation history
if st.session_state['flowmessages']:
    for message in st.session_state['flowmessages']:
        if message['role'] == 'user':
            st.markdown(f"**🧑 You:** {message['content']} (Sentiment: {message['sentiment']}, Intent: {message['intent']})")
        elif message['role'] == 'assistant':
            st.markdown(f"**🤖 Assistant:** {message['content']}")
