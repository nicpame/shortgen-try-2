import os
from transcription.groq_transcription import gen_transcription
from transcription.normalize_tts_files import normalize_transcription
import db.db as db
from config.config import config
cfg= config()


def batch_gen_transcription(vids):

    for vid in vids:
        vid_lang = vid.get('language', 'en')
        res = gen_transcription(vid['source_audio_file_path'], language_code=vid_lang, timestamp_granularities = ['segment'])

        if res and res['text']:
            vid['transcription'] = normalize_transcription(raw_transcription=res)

            vid['state'] = cfg['video_state']['transcription_generated']
            db.update_source_vid(vid)
    
    