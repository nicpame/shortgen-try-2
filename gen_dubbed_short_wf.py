# imports
import datetime
import os
import db.db as db
from helpers.video_dir_name_from_id import video_dir_name_from_id
import translate.translate as translate
from transform import transform
from tts import gen_translated_tc_speech
# config
from config.config import config
cfg = config()

# gen config
gen_cfg = cfg["gen_config"]
video_states_config = cfg["video_state"]

def init_gened_vid(source_vid_id: int):
    """
    Initializes a new entry in the gened_vids_db for a given source video ID.
    """

    # TODO apply gen_config to the new entry 
    gened_vid = {
        "source_vid_id": source_vid_id,
        "state": video_states_config["init"],
        "created_at": datetime.datetime.now().isoformat(),
        # "language": gen_cfg["language"],
        # "transform_layers": gen_cfg["transform_layers"]
    }
    
    gened_vid_id = db.gened_vids_db.insert(gened_vid)
    print(f"Initialized gened_vid entry for source_vid_id: {source_vid_id}")

    # create a folder for this vid if it doesn't exist + update vid in db with created folder dir
    gened_vid_dir = os.path.join(cfg["gened_vids_rel_dir"], video_dir_name_from_id(gened_vid_id))
    if not os.path.exists(gened_vid_dir):
        os.makedirs(gened_vid_dir)
    print(f"Created directory for gened vid: {gened_vid_dir}")
    # update the gened_vid with the directory path
    gened_vid['vid_dir'] = gened_vid_dir
    db.update_gened_vid_by_id(gened_vid_id, gened_vid)



# =============================
# gened vids process steps
# =============================

def process_gened_vid(gened_vid_id: int):
    vid = db.get_gened_vid_by_id(gened_vid_id)
    if not vid:
        print(f"Gened video with id {gened_vid_id} not found.")
        return
    print(f"Processing gened video for gened vid id: {gened_vid_id} with state: {vid['state']}")

    # =============================
    # step 1 : translate transcription if state= "init"
    # =============================
    if vid['state'] == video_states_config["init"]:
        translate_transcription(vid)
    
    # =============================
    # step 1.1 : modify translation if state= "translated" and informalize is enabled
    # =============================
    if vid['state'] == video_states_config["translated"] and vid['gen_config']['transform_layers']['informalize']:
        informalize_translation(vid)

    # =============================
    # step 1.2 : modify translation if state= "translated" and informalize is enabled
    # =============================
    if vid['state'] == video_states_config["informalized"] and vid['gen_config']['transform_layers']['vocalize']:
        vocalize_script(vid)

    # =============================
    # step 2 : gen speech for transformed script
    # =============================
    if vid['state'] == video_states_config["vocalized"]:
        gen_speech(vid)


def translate_transcription(vid: dict):
    print(f"Translating transcription for gened vid id: {vid.doc_id}")
    source_vid = db.get_source_vid_by_id(vid["source_vid_id"])
    if not source_vid or "transcription" not in source_vid:
        print(f"Source video or transcription not found for source_vid_id: {vid['source_vid_id']}")
        return
    transcription = source_vid["transcription"]

    translate_result = translate.translate_transcription(
        normalized_transcription=transcription,
        translation_prompt_file_dir=vid['gen_config']['transform_layers']['translate']['prompt_file_dir'],
        target_language_code=vid['gen_config']['language']
    )

    if translate_result['success']:
        print(f"Translation successful for gened vid id: {vid.doc_id}")
        # Update the gened_vid state to "translated"
        vid['translation'] = translate_result['data']
        vid['state'] = video_states_config["translated"]
        db.update_gened_vid_by_id(vid.doc_id, vid)


def informalize_translation(vid: dict):
    print(f"Informalizing translation for gened vid id: {vid.doc_id}")


    informalize_result = transform.informalize_translation(
        translation=vid['translation'],
        informalize_prompt_file_dir=vid['gen_config']['transform_layers']['informalize']['prompt_file_dir'],
        target_language_code=vid['gen_config']['language'],
        variables=vid['gen_config']['transform_layers']['informalize']['variables']
    )

    if informalize_result['success']:
        print(f"informalize successful for gened vid id: {vid.doc_id}")
        # Update the gened_vid state to "translated"
        vid['informalization'] = informalize_result['data']
        vid['state'] = video_states_config["informalized"]
        db.update_gened_vid_by_id(vid.doc_id, vid)


def vocalize_script(vid: dict):
    print(f"Vocalizing script for gened vid id: {vid.doc_id}")


    vocalize_result = transform.vocalize(
        transformed_script=vid['informalization'],
        vocalize_prompt_file_dir=vid['gen_config']['transform_layers']['vocalize']['prompt_file_dir'],
        variables=vid['gen_config']['transform_layers']['vocalize']['variables']
    )

    if vocalize_result['success']:
        print(f"Vocalize successful for gened vid id: {vid.doc_id}")
        # Update the gened_vid state to "vocalized"
        vid['vocalization'] = vocalize_result['data']
        vid['state'] = video_states_config["vocalized"]
        db.update_gened_vid_by_id(vid.doc_id, vid)

    

def gen_speech(vid: dict):
    print(f"Generating speech for gened vid id: {vid.doc_id}")

    gen_translated_tc_speech.gen_speech(
        translated_tc_dict=vid['vocalization'],
        target_dir=vid['vid_dir'],
        audio_filename=cfg["gened_speech_file_name"])
    # Update the gened_vid state to "speech_generated"
    





# main entry point

if __name__ == "__main__":

    # Initialize gened_vids for each source_vid_id in the config
    # for source_vid_id in gen_cfg["source_vid_ids"]:
    #     init_gened_vid(source_vid_id)

    for gened_vid in db.get_gened_vids():
        vid_id = gened_vid.doc_id

        process_gened_vid(vid_id)
        # Uncomment the above line when process_gened_vid is implemented
