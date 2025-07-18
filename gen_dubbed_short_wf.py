# imports
import datetime
import os
import db.db as db
from helpers.video_dir_name_from_id import video_dir_name_from_id
import translate.translate as translate
from transform import transform
from tts import gen_translated_tc_speech
from tts.adjust_gened_speech_duration import adjust_audio_duration
from metadata.gen_thumbnail import generate_thumbnail
from metadata.gen_metadata import gen_metadata

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
    gened_vid_dir = os.path.join(
        cfg["gened_vids_rel_dir"], video_dir_name_from_id(gened_vid_id)
    )
    if not os.path.exists(gened_vid_dir):
        os.makedirs(gened_vid_dir)
    print(f"Created directory for gened vid: {gened_vid_dir}")
    # update the gened_vid with the directory path
    gened_vid["vid_dir"] = gened_vid_dir
    db.update_gened_vid_by_id(gened_vid_id, gened_vid)


def process_gened_vid(gened_vid_id: int):
    vid = db.get_gened_vid_by_id(gened_vid_id)
    if not vid:
        print(f"Gened video with id {gened_vid_id} not found.")
        return
    print(
        f"Processing gened video for gened vid id: {gened_vid_id} with state: {vid['state']}"
    )

    # =============================
    # step 1 : translate transcription if state= "init"
    # =============================
    if vid["state"] == video_states_config["init"]:
        translate_transcription(vid)

    # =============================
    # step 1.1 : modify translation if state= "translated" and informalize is enabled
    # =============================
    if (
        vid["state"] == video_states_config["translated"]
        and vid["gen_config"]["transform_layers"]["informalize"]
    ):
        informalize_translation(vid)

    # =============================
    # step 1.2 : modify translation if state= "translated" and informalize is enabled
    # =============================
    if (
        vid["state"] == video_states_config["informalized"]
        and vid["gen_config"]["transform_layers"]["vocalize"]
    ):
        vocalize_script(vid)

    # =============================
    # step 2 : gen speech for transformed script
    # =============================
    if vid["state"] == video_states_config["vocalized"]:
        gen_speech(vid)

    # =============================
    # step 3 : mix adjusted speech to video
    # =============================
    if vid["state"] == video_states_config["speech_adjusted"]:
        mix_audio_to_video(vid)

    # =============================
    # step 4 : gen thumbnail
    # =============================
    if vid["state"] == video_states_config["audio_video_mixed"]:
        gen_thumbnail(vid)

    # =============================
    # step 4 : gen metadata (title and description)
    # =============================
    if vid["state"] == video_states_config["thumbnail_generated"]:
        gen_title_and_description(vid)

# =============================
# gened vids process steps
# =============================


def translate_transcription(vid: dict):
    print(f"Translating transcription for gened vid id: {vid.doc_id}")
    source_vid = db.get_source_vid_by_id(vid["source_vid_id"])
    if not source_vid or "transcription" not in source_vid:
        print(
            f"Source video or transcription not found for source_vid_id: {vid['source_vid_id']}"
        )
        return
    transcription = source_vid["transcription"]

    translate_result = translate.translate_transcription(
        normalized_transcription=transcription,
        translation_prompt_file_dir=vid["gen_config"]["transform_layers"]["translate"][
            "prompt_file_dir"
        ],
        target_language_code=vid["gen_config"]["language"],
    )

    if translate_result["success"]:
        print(f"Translation successful for gened vid id: {vid.doc_id}")
        # Update the gened_vid state to "translated"
        vid["translation"] = translate_result["data"]
        vid["state"] = video_states_config["translated"]
        db.update_gened_vid_by_id(vid.doc_id, vid)


def informalize_translation(vid: dict):
    print(f"Informalizing translation for gened vid id: {vid.doc_id}")

    informalize_result = transform.informalize_translation(
        translation=vid["translation"],
        informalize_prompt_file_dir=vid["gen_config"]["transform_layers"][
            "informalize"
        ]["prompt_file_dir"],
        target_language_code=vid["gen_config"]["language"],
        variables=vid["gen_config"]["transform_layers"]["informalize"]["variables"],
    )

    if informalize_result["success"]:
        print(f"informalize successful for gened vid id: {vid.doc_id}")
        # Update the gened_vid state to "translated"
        vid["informalization"] = informalize_result["data"]
        vid["state"] = video_states_config["informalized"]
        db.update_gened_vid_by_id(vid.doc_id, vid)


def vocalize_script(vid: dict):
    print(f"Vocalizing script for gened vid id: {vid.doc_id}")

    vocalize_result = transform.vocalize(
        transformed_script=vid["informalization"],
        vocalize_prompt_file_dir=vid["gen_config"]["transform_layers"]["vocalize"][
            "prompt_file_dir"
        ],
        variables=vid["gen_config"]["transform_layers"]["vocalize"]["variables"],
    )

    if vocalize_result["success"]:
        print(f"Vocalize successful for gened vid id: {vid.doc_id}")
        # Update the gened_vid state to "vocalized"
        vid["vocalization"] = vocalize_result["data"]
        vid["state"] = video_states_config["vocalized"]
        db.update_gened_vid_by_id(vid.doc_id, vid)


def gen_speech(vid: dict):
    print(f"Generating speech for gened vid id: {vid.doc_id}")

    gen_speech_result = gen_translated_tc_speech.gen_speech(
        translated_tc_dict=vid["vocalization"],
        target_dir=vid["vid_dir"],
        audio_filename=cfg["gened_speech_file_name"],
        voice_name=vid["gen_config"]["speech"]["voice_name"],
    )
    # Update the gened_vid state to "speech_generated"
    if gen_speech_result["success"]:
        print(f"Speech generation successful for gened vid id: {vid.doc_id}")
        vid["speech"] = vid.get("speech", {})
        vid["speech"]["gened_speech_file_dir"] = gen_speech_result[
            "gened_speech_file_dir"
        ]
        vid["state"] = video_states_config["speech_generated"]
        db.update_gened_vid_by_id(vid.doc_id, vid)

        # Adjust the audio duration after speech generation
        target_duration = (
            db.get_source_vid_by_id(vid["source_vid_id"])["metadata"]["video_info"][
                "duration"
            ]
            - 0.5
        )
        adjust_audio_duration(
            input_path=vid["speech"]["gened_speech_file_dir"],
            output_path=os.path.join(vid["vid_dir"], cfg["adjusted_speech_file_name"]),
            target_duration=target_duration,
        )

        vid["speech"]["adjusted_speech_file_dir"] = os.path.join(
            vid["vid_dir"], cfg["adjusted_speech_file_name"]
        )
        vid["state"] = video_states_config["speech_adjusted"]
        db.update_gened_vid_by_id(vid.doc_id, vid)


def mix_audio_to_video(vid: dict):
    """
    Mix the generated audio to the video.
    """
    print(f"Mixing audio to video for gened vid id: {vid.doc_id}")

    from mix_vid.mix_audio_to_vid import mix_wav_to_mp4

    input_video_path = db.get_source_vid_by_id(vid["source_vid_id"]).get(
        "source_vid_file_path"
    )
    input_audio_path = os.path.join(vid["vid_dir"], cfg["adjusted_speech_file_name"])
    output_video_path = os.path.join(vid["vid_dir"], cfg["output_video_file_name"])

    mix_wav_to_mp4(
        input_video_path=input_video_path,
        input_audio_path=input_audio_path,
        output_video_path=output_video_path,
    )

    # update the gened_vid with the output video file path
    vid["output_video_file_dir"] = output_video_path
    vid["state"] = video_states_config["audio_video_mixed"]
    db.update_gened_vid_by_id(vid.doc_id, vid)


def gen_thumbnail(vid: dict):
    
    variation_count = vid['gen_config']['thumbnail']['variation_count']
    vid["thumbnail"] = []
    for variation_index in range(variation_count):
        thumbnail_result = generate_thumbnail(vid, variation_index)
        if thumbnail_result["result"]:
            print(f"Thumbnail generation successful for gened vid id: {vid.doc_id}, variation {variation_index}")
            vid["thumbnail"].append({
                "file_path": thumbnail_result["thumbnail_file_path"],
                "formatted_prompt": thumbnail_result["formatted_prompt"],
                "variation_index": variation_index
            })
    if vid["thumbnail"]:
        vid["state"] = video_states_config["thumbnail_generated"]
        db.update_gened_vid_by_id(vid.doc_id, vid)

def gen_title_and_description(vid: dict):

    vid['metadata'] = {}

    metadata_result  = gen_metadata(vid = vid)
    if metadata_result['success'] :
        vid['metadata']['variations'] = metadata_result['variations']
        vid['metadata']['formatted_prompt'] = metadata_result['formatted_prompt']
        vid['state'] = video_states_config["metadata_generated"]
        db.update_gened_vid_by_id(vid.doc_id, vid)
# main entry point

if __name__ == "__main__":
    # Initialize gened_vids for each source_vid_id in the config
    # for source_vid_id in gen_cfg["source_vid_ids"]:
    #     init_gened_vid(source_vid_id)

    for gened_vid in db.get_gened_vids():
        vid_id = gened_vid.doc_id

        process_gened_vid(vid_id)
        # Uncomment the above line when process_gened_vid is implemented
