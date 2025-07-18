from pprint import pprint

import db.db as db
import dl.dl as dl
from dl.extract_audios_from_vids import batch_extract_audio_from_video
from state.update_state_after_batch_dl import update_state_after_batch_dl
from transcription.transcription import batch_gen_transcription
from translate.translate import batch_translate_transcription
from tts.gen_translated_tc_speech import batch_gen_speech
from tts.adjust_gened_speech_duration import batch_adjust_audio_duration
from mix_vid.mix_audio_to_vid import batch_mix_audio_to_video

# load config
from config.config import config
cfg = config()

#  run  configs
# target_language_code = "fa"

# step 1: Import candidate source vids to the database - source_vids table
# db.import_candidate_source_vids_to_db()
# db.import_candidate_source_vids_txt_to_db()


# step 2: dl vids in source_vids table
dl.dl_batch_vids(db.get_candidate_source_vids())

# step 3: Process the downloaded videos and update the database
# update_state_after_batch_dl(db.get_candidate_source_vids())

# step 4: Extract audios from the downloaded videos
batch_extract_audio_from_video(cfg["source_vids_rel_dir"])

# step 5: Generate transcriptions for the extracted audios + normalize the transcriptions
# batch_gen_transcription(cfg["source_vids_rel_dir"])

# step 6: Translate the normalized transcriptions
# batch_translate_transcription_results = batch_translate_transcription(
#     cfg["source_vids_rel_dir"],
#     cfg["translate_prompt_file_dir"],
#     target_language_code=target_language_code,
# )

# step 7: Generate speech from the translated transcriptions
# batch_gen_speech(cfg["source_vids_rel_dir"])

# step 8: Adjust the generated speech duration to match the video duration
# batch_adjust_audio_duration(cfg["source_vids_rel_dir"])

# step 9: Mix the adjusted speech with the original videos
# batch_mix_audio_to_video(cfg["source_vids_rel_dir"])
