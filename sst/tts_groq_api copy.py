import os
import json
from groq import Groq
from dotenv import load_dotenv  # Add this import

load_dotenv()

# Initialize the Groq client
client = Groq()

# Specify the path to the audio file
filename = os.path.dirname(__file__) + "/YOUR_AUDIO.wav" # Replace with your audio file!
filename = r"C:\Users\pame\Documents\dev\shortgen-try-2\data\dled_shorts\GCbMNVgRrRI\the luckiest man ever_GCbMNVgRrRI.mp3"

# Open the audio file
with open(filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
        file=file, # Required audio file
        model="whisper-large-v3-turbo", # Required model to use for transcription
        response_format="verbose_json",  # Optional
        timestamp_granularities=["segment"], # Optional
        language="en",  # Optional
        temperature=0.0  # Optional
    )

    result = {}
    if hasattr(transcription, "text"):
        result["text"] = transcription.text
    if hasattr(transcription, "words"):
        result["words"] = transcription.words
    if hasattr(transcription, "segments"):
        result["segments"] = transcription.segments

    output_path = os.path.splitext(filename)[0] + "_transcription.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, ensure_ascii=False, indent=2)