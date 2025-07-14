import sys, os, json
import llm.gemini_client as gemini_client
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

from config.config import config
cfg = config()

class TransformedSegment(BaseModel):
    original: str
    transformed: str

class TransformList(BaseModel):
    segments : list[TransformedSegment]




def informalize_translation(translation, informalize_prompt_file_dir ,target_language_code, variables=None):

    result = {}

    informalize_prompt = PromptTemplate.from_file(template_file=informalize_prompt_file_dir)

    if variables:
        informalize_prompt = informalize_prompt.partial(**variables)
    
    target_language_name = cfg['language_name'][target_language_code.strip()]
    informalize_prompt = informalize_prompt.invoke(
        {

        'segments': format_segments_to_text(translation, 'translated'),
        'target_language': target_language_name
        }
    )
    formatted_informalize_prompt = informalize_prompt.to_string()
    print(f"  Requesting translation from LLM...")
    informalize_json = gemini_client.generate_json(
        prompt = formatted_informalize_prompt,
        pydantic_schema = TransformList) 

    translation = json.loads(informalize_json)

    if (translation and 
        'segments' in translation and 
        isinstance(translation['segments'], list) and 
        len(translation['segments']) > 0):
        
        result = {
            'success': True,
            'data': {
                'language': target_language_code,
                **translation,
                'formatted_prompt': formatted_informalize_prompt
            }
        }
        return result
    else:
        return {
            'success': False,
            'error': f"Translation failed for language {target_language_code}. No valid segments returned."
        }


def vocalize(transformed_script, vocalize_prompt_file_dir , variables=None):

    result = {}

    vocalize_prompt = PromptTemplate.from_file(template_file=vocalize_prompt_file_dir)

    if variables:
        vocalize_prompt = vocalize_prompt.partial(**variables)

    vocalize_prompt = vocalize_prompt.invoke(
        {
        'segments': format_segments_to_text(transformed_script, 'transformed'),
        }
    )
    formatted_vocalize_prompt = vocalize_prompt.to_string()
    print(f"  Requesting translation from LLM...")
    vocalize_json = gemini_client.generate_json(
        prompt = formatted_vocalize_prompt,
        pydantic_schema = TransformList
    )

    vocalization = json.loads(vocalize_json)

    if (vocalization and 
        'segments' in vocalization and 
        isinstance(vocalization['segments'], list) and 
        len(vocalization['segments']) > 0):
        
        result = {
            'success': True,
            'data': {
                **vocalization,
                'formatted_prompt': formatted_vocalize_prompt
            }
        }
        return result
    else:
        return {
            'success': False,
            'error': f"vocalization failed for language . No valid segments returned."
        }






def format_segments_to_text(segmented_script: dict, sub_segment_key_name: str) -> str:
    
    segments_data = segmented_script["segments"]
    if not segments_data:
        return "<segments></segments>"
    
    segments = ["<segments>"]
    
    for segment in segments_data:
        if isinstance(segment, dict) and sub_segment_key_name in segment:
            segments.append(f"<segment>{segment[sub_segment_key_name]}</segment>")
    segments.append("</segments>")
    
    return "\n".join(segments)