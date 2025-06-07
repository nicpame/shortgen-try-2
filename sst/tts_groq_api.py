import os
import json
from groq import Groq
from dotenv import load_dotenv  # Add this import

load_dotenv()  #

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
      timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
      language="en",  # Optional
      temperature=0.0  # Optional
    )
    # To print only the transcription text, you'd use print(transcription.text) (here we're printing the entire transcription object to access timestamps)
    print(json.dumps(transcription, indent=2, default=str))
    # Write the transcription JSON to a text file
    output_path = os.path.splitext(filename)[0] + "_transcription.txt"
    with open(output_path, "w", encoding="utf-8") as out_file:
      out_file.write(json.dumps(transcription, indent=2, default=str))