# imports
import datetime
import db.db as db

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
    
    db.gened_vids_db.insert(gened_vid)
    print(f"Initialized gened_vid entry for source_vid_id: {source_vid_id}")




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
        print(f"Translating transcription for gened vid id: {gened_vid_id}")
        translate_transcription(vid)
    

def translate_transcription(vid: dict):
    source_vid = db.get_source_vid_by_id(vid["source_vid_id"])
    if not source_vid or "transcription" not in source_vid:
        print(f"Source video or transcription not found for source_vid_id: {vid['source_vid_id']}")
        return
    transcription = source_vid["transcription"]
    # translated = db.translate(transcription)
    # print(f"Translated transcription for gened vid id {vid['source_vid_id']}: {translated}")



# main entry point

if __name__ == "__main__":

    # Initialize gened_vids for each source_vid_id in the config
    # for source_vid_id in gen_cfg["source_vid_ids"]:
    #     init_gened_vid(source_vid_id)

    for gened_vid in db.get_gened_vids():
        vid_id = gened_vid.doc_id

        process_gened_vid(vid_id)
        # Uncomment the above line when process_gened_vid is implemented
