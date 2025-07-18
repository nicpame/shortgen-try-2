import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

def gen_transcription(audio_file_dir, language_code = 'en' , timestamp_granularities = ['segment']):
    print(f"Generating transcription for: {audio_file_dir}")
    # Open the audio file
    with open(audio_file_dir, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
            file=file, # Required audio file
            model="whisper-large-v3-turbo", # Required model to use for transcription
            response_format="verbose_json",  # Optional
            timestamp_granularities= timestamp_granularities, # Optional can be ['word'] ['segment'] ['segment', 'word']
            language=language_code,  # Optional
            temperature=0.0  # Optional
        )

        result = {}
        if hasattr(transcription, "text"):
            result["text"] = transcription.text
            result['language'] = language_code
        if hasattr(transcription, "words"):
            result["words"] = transcription.words
        if hasattr(transcription, "segments"):
            result["segments"] = transcription.segments

        return result