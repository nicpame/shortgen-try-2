import os, json
from typing import Dict, Union


def read_prompt_template(file_path: str) -> str:
    """
    Read the prompt template from a markdown file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading prompt template file: {e}")


def format_transcription_segments(sub: Dict) -> str:
    """
    Format the transcription dictionary into XML-like segment format.
    """
    if not sub or "segments" not in sub:
        return "<segments></segments>"
    
    segments_data = sub["segments"]
    if not segments_data:
        return "<segments></segments>"
    
    segments = ["<segments>"]
    
    for segment in segments_data:
        if isinstance(segment, dict) and "text" in segment:
            segments.append(f"<segment>{segment['text']}</segment>")
    segments.append("</segments>")
    
    return "\n".join(segments)

def cook_prompt(prompt_template: str, sub: Dict, target_language: str) -> str:
    """
    Process the prompt template by replacing variables with actual values.
    """
    if not prompt_template:
        raise ValueError("Prompt template cannot be empty")
    
    if not target_language or not target_language.strip():
        raise ValueError("Target language must be specified")
    
    if not isinstance(sub, dict):
        raise ValueError("Transcription data must be a dictionary")
    
    # Format the transcription segments
    transcription_segments = format_transcription_segments(sub)
    
    # Replace template variables
    cooked_prompt = prompt_template
    cooked_prompt = cooked_prompt.replace("{TARGET_LANGUAGE}", target_language.strip())
    cooked_prompt = cooked_prompt.replace("{TRANSCRIPTION_SEGMENTS}", transcription_segments)
    
    return cooked_prompt

def process_translation_prompt(prompt_file_path: str, sub: Dict, target_language: str) -> str:

    # Read the prompt template from file
    prompt_template = read_prompt_template(prompt_file_path)
    
    # Cook the prompt with provided parameters
    cooked_prompt = cook_prompt(prompt_template, sub, target_language)
    
    return cooked_prompt



def batch_cook_translate_prompt(videos_dir, prompt_file_path, target_language):
    abs_path = os.path.abspath(videos_dir)  # Convert relative path to absolute

    if not os.path.isdir(abs_path):  # Ensure it's a valid directory
        print(f"Error: {abs_path} is not a directory")
        return

    for subfolder in os.listdir(abs_path):  # List items in the directory
        subfolder_path = os.path.join(abs_path, subfolder)
        
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            target_file_path = os.path.join(subfolder_path, NORMALIZED_TRANSCRIPTION_FILE_NAME)
            
            if os.path.isfile(target_file_path):  # Check if the file exists
                with open(target_file_path, 'r', encoding='utf-8') as file:
                    sub_data = json.load(file)
                    
                # Process the translation prompt with the loaded JSON data
                translation_prompt = process_translation_prompt(
                    prompt_file_path=prompt_file_path,
                    sub=sub_data,
                    target_language=target_language
                )
                # Write translation prompt to JSON file
                output_file = os.path.join(subfolder_path, "translate_prompt.json")
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({"prompt": translation_prompt}, f, ensure_ascii=False, indent=2)
                except IOError as e:
                    print(f"Error writing translation prompt to {output_file}: {e}")



# config
NORMALIZED_TRANSCRIPTION_FILE_NAME = "normalized_transcription.json"
VIDEOS_DIR = "./data/dled_shorts"
TRANSLATE_PROMPT_DIR = './prompts/translate_transcription.md'
TARGET_LANG = "Persian"

# Example usage and testing
if __name__ == "__main__":

    batch_cook_translate_prompt(
        videos_dir = VIDEOS_DIR,
        prompt_file_path=TRANSLATE_PROMPT_DIR,
        target_language=TARGET_LANG
    )
        
