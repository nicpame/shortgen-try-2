import os
from dotenv import load_dotenv
from google import genai
load_dotenv()

GEMINI_MODELS = {
    "flash": "gemini-2.5-flash-preview-05-20",
    "pro": "gemini-1.5-pro-latest",
    "pro-vision": "gemini-1.5-pro-vision-latest"
}
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def generate_text(prompt: str, model_key: str = "flash") -> str:
    global client
    
    model_id = GEMINI_MODELS.get(model_key, GEMINI_MODELS["flash"])
    response = client.models.generate_content(
        model=model_id,
        contents=prompt
    )
    return response.text

def generate_json(prompt: str, pydantic_schema, model_key: str = "flash") -> str:
    global client
    
    model_id = GEMINI_MODELS.get(model_key, GEMINI_MODELS["flash"])
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": pydantic_schema,
        },
    )
    return response.text


    