import os
from moviepy import VideoFileClip
import db.db as db
from config.config import config
cfg = config()


def process_video(vid):
    """Extracts audio from a video file and saves it in the same directory with the same name"""

    source_vid_path = vid['source_vid_file_path']
    try:
        video = VideoFileClip(source_vid_path)
        audio = video.audio

        if audio:
            parent_dir = os.path.dirname(source_vid_path)
            audio_file_path = os.path.join(parent_dir, cfg['source_audio_file_name'])
            audio.write_audiofile(audio_file_path, bitrate="120k")
            print(f"Audio saved: {audio_file_path}")
            
            # update db item
            vid['state'] = cfg['video_state']['audio_extracted']
            vid['source_audio_file_path'] = audio_file_path
            db.update_source_vid(vid)
        
        video.close()
    except Exception as e:
        print(f"Error processing {source_vid_path}: {e}")



def batch_extract_audio_from_video(vids):

    for vid in vids:
        process_video(vid)
