import sys, os, json
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.gemini_client import generate_speech_and_save_file
from helpers.word_count import count_words
import wave

AUDIO_FILE_NAME="gened_speech.wav"
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


def concat_wav_files(parts_paths, output_path):
    """
    Concatenate multiple wav files into one output file.
    Assumes all wav files have the same parameters (channels, sample width, framerate).
    """
    if not parts_paths:
        return
    with wave.open(parts_paths[0], 'rb') as wf:
        params = wf.getparams()
        frames = [wf.readframes(wf.getnframes())]
    for part_path in parts_paths[1:]:
        with wave.open(part_path, 'rb') as wf:
            frames.append(wf.readframes(wf.getnframes()))
    with wave.open(output_path, 'wb') as wf_out:
        wf_out.setparams(params)
        for f in frames:
            wf_out.writeframes(f)


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

    part_paths = []
    for i, part in enumerate(tcs_parts):
        base_filename = os.path.splitext(audio_filename)[0]
        part_filename = f"{base_filename}_part_{i}.wav"
        part_path = os.path.join(audio_file_path, part_filename)
        generate_speech_and_save_file(
            text=part,
            audio_file_path=audio_file_path,
            audio_filename=part_filename
        )
        part_paths.append(part_path)

    # Join all parts into one file
    joined_audio_path = os.path.join(audio_file_path, audio_filename)
    concat_wav_files(part_paths, joined_audio_path)
    
batch_gen_speech(VIDEOS_DIR)


