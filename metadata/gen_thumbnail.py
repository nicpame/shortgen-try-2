from langchain.prompts import PromptTemplate
import os
# config
from config.config import config
cfg = config()


# Assume gen_image_with_gemini is defined elsewhere and imported here
from llm.gemini_client import generate_image_and_save_file

def generate_thumbnail(vid_title, vid_description, vid_overlay_text, target_language, vid_dir,  prompt_file_path):
    """
    Generates a thumbnail image using Gemini based on a formatted prompt.

    Args:
        vid_title (str): The video title.
        vid_description (str): The video description.
        vid_overlay_text (str): The overlay text for the thumbnail.
        prompt_file_path (str): Path to the prompt template file.

    Returns:
        The generated image (format depends on gen_image_with_gemini).
    """
    # Create a LangChain PromptTemplate from file
    prompt = PromptTemplate.from_file(prompt_file_path)

    # Format the prompt with the provided variables
    formatted_prompt = prompt.format(
        video_title=vid_title,
        video_description=vid_description,
        overlay_text=vid_overlay_text,
        target_language=target_language
    )

    # Generate the image using Gemini
    res = generate_image_and_save_file(
        prompt=formatted_prompt,
        image_file_path= os.path.join(vid_dir, cfg["thumbnail_file_name"])
    )

    #  if res return {result, "image_path": res} else return false
    if res:
        return {"result": True, "thumbnail_file_path": res , "formatted_prompt": formatted_prompt}
    else:
        return {"result": False, "error": "Image generation failed."}