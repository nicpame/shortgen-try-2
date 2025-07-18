import os
import json



def normalize_transcription(raw_transcription):
    """
    Reads a JSON file, transforms its structure, and writes the normalized JSON to:
      normalized_transcription.json (in the same directory)
    """
    new_data = {}

    
    # Rename the top-level "text" key to "transcription"
    if "text" in raw_transcription:
        new_data["transcription"] = raw_transcription.get("text")
    
    # Process segments if they exist and are a list
    if "segments" in raw_transcription and isinstance(raw_transcription["segments"], list):
        new_data["segments"] = []
        for segment in raw_transcription["segments"]:
            # Calculate duration using segment's "start" and "end" values.
            new_segment = {}
            try:
                start = float(segment.get("start", 0))
                end = float(segment.get("end", 0))
                new_segment['start'] = start
                new_segment['end'] = end
                new_segment["duration"] = round(end - start, 2)
            except (ValueError, TypeError):
                new_segment["duration"] = None  # or handle error as needed
                
            # Copy the segment's "text" value to a new key "transcription"
            if "text" in segment:
                new_segment["text"] = segment["text"]

            new_data["segments"].append(new_segment)
    
    return new_data

# Example usage
# batch_normalize_transcriptions(VIDEOS_DIR)