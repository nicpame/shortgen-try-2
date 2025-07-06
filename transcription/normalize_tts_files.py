import os
import json


def batch_normalize_transcriptions(source_vids_rel_dir, raw_transcription_file_name, normalized_transcription_file_name):
    """ Scans only the direct subfolders inside the given directory for the target file """
    abs_path = os.path.abspath(source_vids_rel_dir)  # Convert relative path to absolute

    if not os.path.isdir(abs_path):  # Ensure it's a valid directory
        print(f"Error: {abs_path} is not a directory")
        return

    for subfolder in os.listdir(abs_path):  # List items in the directory
        subfolder_path = os.path.join(abs_path, subfolder)
        
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            target_file_path = os.path.join(subfolder_path, raw_transcription_file_name)
            
            if os.path.isfile(target_file_path):  # Check if the file exists
                normalize_transcription(target_file_path, normalized_transcription_file_name)  # Pass file path to processing function




def normalize_transcription(json_file_path, normalized_transcription_file_name):
    """
    Reads a JSON file, transforms its structure, and writes the normalized JSON to:
      normalized_transcription.json (in the same directory)
    """
    new_data = {}
    # Load the original JSON data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Rename the top-level "text" key to "transcription"
    if "text" in data:
        new_data["transcription"] = data.get("text")
    
    # Process segments if they exist and are a list
    if "segments" in data and isinstance(data["segments"], list):
        new_data["segments"] = []
        for segment in data["segments"]:
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
    
    # Write the transformed JSON to a new file in the same directory
    output_path = os.path.join(os.path.dirname(json_file_path), normalized_transcription_file_name)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(new_data, out_file, indent=2, ensure_ascii=False)
    
    print(f"Transformed file written to: {output_path}")

# Example usage
# batch_normalize_transcriptions(VIDEOS_DIR)