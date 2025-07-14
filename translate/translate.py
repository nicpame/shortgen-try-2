import sys, os, json
import llm.gemini_client as gemini_client
from translate.cook_translate_prompts import get_translation_prompt
from pydantic import BaseModel

from config.config import config
cfg = config()

class TranslatedSegment(BaseModel):
    original: str
    translated: str

class TranslationList(BaseModel):
    segments : list[TranslatedSegment]






def batch_translate_transcription(videos_dir, translation_prompt_file_dir ,target_language_code="fa"):

    results = [] 

    print(f"Starting batch translation in directory: {videos_dir}")

    for subfolder in os.listdir(videos_dir):  # List items in the directory
        print(f"Processing video folder: {subfolder}")
        result = {
            'vid_id': int(subfolder.strip()),  # Assuming the subfolder name starts with vid_id
            'translations': {}
        }
        vid_dir = os.path.join(videos_dir, subfolder)

        normalized_transcription_path = os.path.join(vid_dir, cfg['normalized_transcription_file_name'])
        print(f"  Loading transcription: {normalized_transcription_path}")
        with open(normalized_transcription_path, 'r', encoding='utf-8') as f:
            normalized_transcription = json.load(f)

        print(f"  Generating translation prompt for language: {target_language_code}")
        translation_prompt = get_translation_prompt(
            prompt_file_path = translation_prompt_file_dir, 
            normalized_transcription = normalized_transcription, 
            target_language_code = target_language_code)
        
    
        print(f"  Requesting translation from LLM...")
        translation_json = gemini_client.generate_json(
            prompt = translation_prompt,
            pydantic_schema = TranslationList) 
        
        translation = json.loads(translation_json)

        # Store the translation in a dictionary with the language code as the key
        result['translations'][target_language_code] = translation
        
        results.append(result)

        output_file = os.path.join(vid_dir, "translated_transcritption.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({'translations': result['translations']}, f, ensure_ascii=False, indent=2)
            print(f"  Saved translation to: {output_file}")
        except IOError as e:
            print(f"Error writing translation prompt to {output_file}: {e}")

    print("Batch translation complete.")
    return results




def translate_transcription(normalized_transcription, translation_prompt_file_dir ,target_language_code="fa"):

    result = {}

    translation_prompt = get_translation_prompt(
        prompt_file_path = translation_prompt_file_dir, 
        normalized_transcription = normalized_transcription, 
        target_language_code = target_language_code)
    

    print(f"  Requesting translation from LLM...")
    translation_json = gemini_client.generate_json(
        prompt = translation_prompt,
        pydantic_schema = TranslationList) 

    translation = json.loads(translation_json)

    if (translation and 
        'segments' in translation and 
        isinstance(translation['segments'], list) and 
        len(translation['segments']) > 0):
        
        result = {
            'success': True,
            'data': {
                'language': target_language_code,
                **translation
            }
        }
        return result
    else:
        return {
            'success': False,
            'error': f"Translation failed for language {target_language_code}. No valid segments returned."
        }




