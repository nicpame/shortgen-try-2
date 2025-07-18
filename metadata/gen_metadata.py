
from pydantic import BaseModel
import db.db as db

from langchain.prompts.loading import load_prompt
from llm.gemini_client import generate_json

class VideoMetadata(BaseModel):
    video_title: str
    video_description: str




def gen_metadata(vid: dict) -> list :
    """
    Given a video dict, generate a list of 4 VideoMetadata objects using Gemini and a prompt template.
    """
    source_vid = db.get_source_vid_by_id(vid["source_vid_id"])
    vid_title = source_vid["metadata"]["video_info"]["title"]
    vid_description = source_vid["metadata"]["video_info"]["description"]
    target_language = vid["gen_config"]['language']
    vid_dir = vid["vid_dir"]
    prompt_path = vid["gen_config"]['metadata']['title_and_description_prompt_path']
    variation_count = vid["gen_config"]['metadata']['variation_count']

    # Load and format the prompt using langchain
    prompt_template = load_prompt(prompt_path)
    prompt = prompt_template.format(
        video_title=vid_title,
        video_description=vid_description,
        target_language = target_language,
        variation_count = variation_count
    )


    response_json = generate_json(prompt, pydantic_schema=list[VideoMetadata])

    import json
    data = json.loads(response_json)
    
    if data and len(data) > 0 :
        return {
            "success" : True,
            "variations": data,
            "formatted_prompt": prompt
        }
    else :
        return {
            "success" : False,
            "message": "metadata gen failed"
        }
