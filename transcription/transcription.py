import os
from transcription.groq_transcription import gen_transcription_and_write
from transcription.normalize_tts_files import batch_normalize_transcriptions
from config.config import config
cfg= config()


def batch_gen_transcription(source_vids_rel_dir):
    for folder_name in os.listdir(source_vids_rel_dir):
        folder_path = os.path.join(source_vids_rel_dir, folder_name)

        if os.path.isdir(folder_path):  # Check if it's a subfolder
            audio_file = None
            has_transcription = False
            
            # Check for audio file that does not have a transcription already
            for file in os.listdir(folder_path):
                if 'source_audio' in file:
                    audio_file = os.path.join(folder_path, file)
                elif cfg['raw_transcription_file_name'] in file:
                    has_transcription = True
            
            if audio_file and not has_transcription:
                gen_transcription_and_write(audio_file)
    
    # Normalize the generated transcriptions to have a consistent and simpler structure
    batch_normalize_transcriptions(
        source_vids_rel_dir=source_vids_rel_dir,
        raw_transcription_file_name=cfg['raw_transcription_file_name'],
        normalized_transcription_file_name=cfg['normalized_transcription_file_name'])