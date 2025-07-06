import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

def gen_transcription_and_write(audio_file_dir):
    print(f"Generating transcription for: {audio_file_dir}")
    # Open the audio file
    with open(audio_file_dir, "rb") as file:
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

        parent_dir = os.path.dirname(audio_file_dir)
        output_path = os.path.join(parent_dir, "source_transcription.json")
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(result, out_file, ensure_ascii=False, indent=2)