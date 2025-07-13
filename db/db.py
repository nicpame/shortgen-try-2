from tinydb import TinyDB, Query
import json, os
from helpers.video_dir_name_from_id import video_dir_name_from_id

from config.config import config
cfg = config()


DB = "./data/db/db.json"
CANDIDATE_SOURCE_VIDS = "./data/candidate_source_vids.json"
CANDIDATE_SOURCE_VIDS = "data/candidate_source_vids.txt"

db = TinyDB(DB, indent=4)
source_vids_db = db.table("source_vids")
gened_vids_db = db.table("gened_vids")

video_states_config = cfg["video_state"]


def import_candidate_source_vids_to_db():
    with open(CANDIDATE_SOURCE_VIDS, "r", encoding="utf-8") as f:
        data = json.load(f)

    inserted_count = 0
    for item in data:
        url = item.get("url")
        if url is None:
            continue  # skip items without a url
        if not source_vids_db.search(Query().url == url):
            item["state"] = video_states_config["candidate"]
            source_vids_db.insert(item)
            inserted_count += 1
    print(f"Inserted {inserted_count} new items into the source_vids_db table.")


def get_candidate_source_vids():
    return source_vids_db.search(Query().state == video_states_config["candidate"])

def get_source_vid_by_id(source_vid_id: int):

    source_vid = source_vids_db.get(doc_id=source_vid_id)
    if source_vid is None:
        return None

    source_vids_dir = cfg["source_vids_rel_dir"]

    vid_dir = os.path.join(source_vids_dir, video_dir_name_from_id(source_vid_id))



    # TODO remove this logic when these info is added when dl process happens
    # get source vid file path if it exists
    source_vid_file_path = os.path.join(vid_dir, cfg["source_vid_file_name"])
    if os.path.isfile(source_vid_file_path) and not source_vid.get("source_vid_file_path"):
        source_vid["source_vid_file_path"] = source_vid_file_path
    

    # get source audio file path if it exists
    source_audio_file_path = os.path.join(vid_dir, cfg["source_audio_file_name"])
    if os.path.isfile(source_audio_file_path) and not source_vid.get("source_audio_file_path"):
        source_vid["source_audio_file_path"] = source_audio_file_path

    # get source thumbnail file path if it exists
    source_thumbnail_file_path = os.path.join(vid_dir, cfg["source_thumbnail_file_name"])
    if os.path.isfile(source_thumbnail_file_path) and not source_vid.get("source_thumbnail_file_path"):
        source_vid["source_thumbnail_file_path"] = source_thumbnail_file_path

    # get source metadata file path if it exists
    source_metadata_file_path = os.path.join(vid_dir, cfg["source_metadata_file_name"])
    if os.path.isfile(source_metadata_file_path) and not source_vid.get("metadata"):
        # read metadata from file and add it to the source_vid dict
        with open(source_metadata_file_path, 'r', encoding='utf-8') as f:
            try:
                metadata = json.load(f)
                source_vid["metadata"] = metadata
            except json.JSONDecodeError:
                print(f"Error decoding JSON from metadata file: {source_metadata_file_path}")
                source_vid["metadata"] = {}

    # get source normalized transcription file path if it exists and set it in source_vid
    source_normalized_transcription_file_path = os.path.join(vid_dir, cfg["normalized_transcription_file_name"])
    if os.path.isfile(source_normalized_transcription_file_path) and not source_vid.get("transcription"):
        # read normalized transcription from file and add it to the source_vid dict
        with open(source_normalized_transcription_file_path, 'r', encoding='utf-8') as f:
            try:
                normalized_transcription = json.load(f)
                source_vid["transcription"] = normalized_transcription
            except json.JSONDecodeError:
                print(f"Error decoding JSON from normalized transcription file: {source_normalized_transcription_file_path}")
                source_vid["transcription"] = {}

    update_source_vids_in_db_batch([source_vid])
    return source_vid


def update_source_vids_in_db_batch(updated_source_vids):
    updated_count = 0
    for item in updated_source_vids:
        url = item.get("url")
        if not url:
            continue
        result = source_vids_db.update(item, Query().url == url)
        if result:
            updated_count += 1
    print(f"Batch updated {updated_count} entries in the source_vids_db table.")


def import_candidate_source_vids_txt_to_db(txt_path=CANDIDATE_SOURCE_VIDS):
    """
    Imports candidate source video URLs from a plain text file (one URL per line)
    into the source_vids_db table if not already present.
    """
    inserted_count = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            if not source_vids_db.search(Query().url == url):
                item = {"url": url, "state": video_states_config["candidate"]}
                source_vids_db.insert(item)
                inserted_count += 1
    print(
        f"Inserted {inserted_count} new items from text file into the source_vids_db table."
    )


# =============================
# gened_vids_db functions
# =============================
def get_gened_vids():
    return gened_vids_db.all()

def get_gened_vid_by_source_vid_id(source_vid_id: int):
    """
    Retrieves a gened video entry by its source video ID.
    """
    return gened_vids_db.search(Query().source_vid_id == source_vid_id)

def get_gened_vid_by_id(vid_id: int):
    """
    Retrieves a gened video entry by its TinyDB doc_id.
    """
    return gened_vids_db.get(doc_id=vid_id)