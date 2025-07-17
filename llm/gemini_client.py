
from io import BytesIO
from PIL import Image

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODELS = {
    "flash": "gemini-2.5-flash-preview-05-20",
    "pro": "gemini-2.5-pro-preview-06-05",
    "pro-vision": "gemini-1.5-pro-vision-latest",
    "flash-tts": "gemini-2.5-flash-preview-tts",
    "flash-image": "gemini-2.0-flash-preview-image-generation",
}

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def generate_text(prompt: str, model_key: str = "flash") -> str:
    global client

    model_id = GEMINI_MODELS.get(model_key, GEMINI_MODELS["flash"])
    response = client.models.generate_content(model=model_id, contents=prompt)
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


def generate_speech_and_save_file(
    prompt: str,
    audio_file_path: str,
    audio_filename: str,
    gemini_tts_voice_name: str = "Kore",
    model_key: str = "flash-tts",
):
    global client

    model_id = GEMINI_MODELS.get(model_key)

    # Set up the wave file to save the output:
    import wave

    def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)

    # Set up the client:
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=gemini_tts_voice_name,
                    )
                )
            ),
        ),
    )

    data = response.candidates[0].content.parts[0].inline_data.data

    full_path = os.path.join(audio_file_path, audio_filename)
    wave_file(full_path, data)  # Save the file to specified path with given filename


def generate_image_and_save_file(prompt: str, image_file_path: str, model_key: str = "flash-image"):

    """
    Generate an image using Gemini API and save it to a file.
    Args:
        prompt (str): The prompt for image generation.
        file_dir (str): Directory to save the image file.
        image_filename (str): Name of the image file (default: 'gemini_image.png').
    """
    global client

    model_id = GEMINI_MODELS.get(model_key)
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )
    )

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            os.makedirs(image_file_path, exist_ok=True)
            image.save(image_file_path)
            return image_file_path
    return None


def list_available_gemini_models():
    """
    List available Gemini models and print their IDs and supported modalities.
    """
    print("List of models that support generateContent:\n")
    for m in client.models.list():
        if hasattr(m, 'supported_actions'):
            for action in m.supported_actions:
                if action == "generateContent":
                    print(m.name)

    print("\nList of models that support embedContent:\n")
    for m in client.models.list():
        if hasattr(m, 'supported_actions'):
            for action in m.supported_actions:
                if action == "embedContent":
                    print(m.name)