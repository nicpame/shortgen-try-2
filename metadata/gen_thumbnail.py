import os
import db.db as db
from langchain.prompts import PromptTemplate
from llm.gemini_client import generate_image_and_save_file
# config
from config.config import config
cfg = config()


# Assume gen_image_with_gemini is defined elsewhere and imported here

def generate_thumbnail(vid, variation_index):

    source_vid = db.get_source_vid_by_id(vid["source_vid_id"])
    vid_title = source_vid["metadata"]["video_info"]["title"]
    vid_description = source_vid["metadata"]["video_info"]["description"]
    vid_dir = vid["vid_dir"]
    prompt_file_path=vid["gen_config"]["thumbnail"]["prompt_file_dir"]


    # Create a LangChain PromptTemplate from file
    prompt = PromptTemplate.from_file(prompt_file_path)

    # Format the prompt with the provided variables
    formatted_prompt = prompt.format(
        video_title=vid_title,
        video_description=vid_description,
    )

    # Generate the image using Gemini
    base, ext = os.path.splitext(cfg["thumbnail_file_name"])
    thumbnail_file_name = f"{base}[var_{variation_index}]{ext}"

    res = generate_image_and_save_file(
        prompt=formatted_prompt,
        image_file_path= os.path.join(vid_dir, thumbnail_file_name)
    )

    #  if res return {result, "image_path": res} else return false
    if res:
        return {"result": True, "thumbnail_file_path": res , "formatted_prompt": formatted_prompt}
    else:
        return {"result": False, "error": "Image generation failed."}