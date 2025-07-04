import os, sys
sys.path.append(os.getcwd())
from config.config import config
cfg = config()

import db.db as db

from helpers.video_dir_name_from_id import video_dir_name_from_id

source_vids_rel_dir = cfg['source_vids_rel_dir']


def update_state_after_batch_dl(candidate_vids):
    """
    For each candidate video dict, find its folder (e.g., '001', '002', ...),
    then call functions for files inside each folder.
    """
    for vid in candidate_vids:

        vid_dir = os.path.join(source_vids_rel_dir, video_dir_name_from_id(vid.doc_id))
        if not os.path.isdir(vid_dir):
            print(f"Folder not found: {vid_dir}")
            continue
        # List files in the folder
        files = os.listdir(vid_dir)
        # Call your functions for each file as needed


        if is_nonempty_source_vid_present(vid_dir, files):
            vid['state'] = cfg['video_state']['downloaded']

    db.update_source_vids_in_db_batch(candidate_vids)


def is_nonempty_source_vid_present(vid_dir, files):
    source_vid_path = os.path.join(vid_dir, 'source_vid.mp4')
    return 'source_vid.mp4' in files and os.path.getsize(source_vid_path) > 0