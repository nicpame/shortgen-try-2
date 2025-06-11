import sys, os, json
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from llm.gemini_client import generate_json

from pydantic import BaseModel

class TranslatedSegment(BaseModel):
    original: str
    translated: str

class TranslationList(BaseModel):
    translations: list[TranslatedSegment]




def get_gemini_result_from_prompt(prompt_text: str) -> str:
    """Takes a prompt text, passes it to generate_gemini_response, and returns the result."""




    return generate_json(
        prompt=prompt_text,
        pydantic_schema=TranslationList
    )

def batch_translate(videos_dir):
    abs_path = os.path.abspath(videos_dir)  # Convert relative path to absolute

    for subfolder in os.listdir(abs_path):  # List items in the directory
        subfolder_path = os.path.join(abs_path, subfolder)
        
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            target_file_path = os.path.join(subfolder_path, PROMPT_FILE_NAME)
            
            if os.path.isfile(target_file_path):  # Check if the file exists
                with open(target_file_path, 'r', encoding='utf-8') as file:
                    prompts = json.load(file)
                
                translation_json = get_gemini_result_from_prompt(prompts['prompt'])
                translation = json.loads(translation_json)
                print(translation)
                # Write translation prompt to JSON file
                output_file = os.path.join(subfolder_path, "translated_transcritption.json")
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(translation, f, ensure_ascii=False, indent=2)
                except IOError as e:
                    print(f"Error writing translation prompt to {output_file}: {e}")

PROMPT_FILE_NAME="translate_prompt.json"
VIDEOS_DIR = "./data/dled_shorts"
batch_translate(VIDEOS_DIR)