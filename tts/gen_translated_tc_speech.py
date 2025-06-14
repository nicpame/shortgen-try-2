import sys, os, json
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.gemini_client import generate_speech_and_save_file
from helpers.word_count import count_words

AUDIO_FILE_NAME="gened_sppech.wav"
VIDEOS_DIR = "./data/dled_shorts"
TRANSLATED_TC_FILE_NAME="translated_transcritption.json"


def accumulate_tcs_dict_to_parts(texts, threshold):
    parts = []
    current_part = []
    current_count = 0

    for text in texts:
        wc = count_words(text)
        if current_count + wc > threshold:
            # Commit current part to parts list
            if current_part:
                parts.append("\n".join(current_part))
            # Start a new part
            current_part = [text]
            current_count = wc
        else:
            current_part.append(text)
            current_count += wc

    # Add the last accumulated part if any
    if current_part:
        parts.append("\n".join(current_part))

    return parts


def batch_gen_speech(videos_dir):
    abs_path = os.path.abspath(videos_dir)  # Convert relative path to absolute

    for subfolder in os.listdir(abs_path):  # List items in the directory
        subfolder_path = os.path.join(abs_path, subfolder)
        
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            target_file_path = os.path.join(subfolder_path, TRANSLATED_TC_FILE_NAME)
            
            if os.path.isfile(target_file_path):  # Check if the file exists
                with open(target_file_path, 'r', encoding='utf-8') as file:
                    translated_tc_dict = json.load(file)

                gen_speech(translated_tc_dict, subfolder_path, AUDIO_FILE_NAME)



def gen_speech(translated_tc_dict: dict,
               audio_file_path: str,
               audio_filename: str):
    """
    takes translated_transcritption.json as dict, 
    gens sppechs parts(coz gemini has limits in long texts for tts),
    saves parts to <audio_file_path>/<audio_filename> to join files later

    """

    translated_tcs_list = [seg.get('translated', '') for seg in translated_tc_dict['translations']]
    translated_tcs_text = '\n'.join(translated_tcs_list)
    
    word_count_threshold_for_tts = 100
    tcs_parts = accumulate_tcs_dict_to_parts(translated_tcs_list, word_count_threshold_for_tts)

    print(len(tcs_parts))
    for i, part in enumerate(tcs_parts):
        base_filename = os.path.splitext(audio_filename)[0]
        generate_speech_and_save_file(
            text=part,
            audio_file_path=audio_file_path,
            audio_filename=f"{base_filename}_part_{i}.wav"
        )


batch_gen_speech(VIDEOS_DIR)


