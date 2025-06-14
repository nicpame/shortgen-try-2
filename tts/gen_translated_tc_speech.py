import sys, os, json
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.gemini_client import generate_speech_and_save_file


AUDIO_FILE_NAME="gened_sppech.wav"
VIDEOS_DIR = "./data/dled_shorts"
TRANSLATED_TC_FILE_NAME="translated_transcritption.json"



def batch_gen_speech(videos_dir):
    abs_path = os.path.abspath(videos_dir)  # Convert relative path to absolute

    for subfolder in os.listdir(abs_path):  # List items in the directory
        subfolder_path = os.path.join(abs_path, subfolder)
        
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            target_file_path = os.path.join(subfolder_path, TRANSLATED_TC_FILE_NAME)
            
            if os.path.isfile(target_file_path):  # Check if the file exists
                with open(target_file_path, 'r', encoding='utf-8') as file:
                    translated_tc_dict = json.load(file)
                
                translated_tc_text= '\n'.join(seg.get('translated', '') for seg in translated_tc_dict['translations'])
                # Write translated text to file for reference
                text_file_path = os.path.join(subfolder_path, 'translated_text.txt')
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(translated_tc_text)
                # generate_speech_and_save_file(
                #     text=translated_tc_text, 
                #     audio_file_path=subfolder_path, 
                #     audio_filename=AUDIO_FILE_NAME)


batch_gen_speech(VIDEOS_DIR)