import sys, os
import shutil
from dl.youtube_downloader import YouTubeDownloader
import db.db as db

from config.config import config
import requests
cfg = config()

# config
source_vids_dir = cfg['source_vids_rel_dir']

yt_downloader = YouTubeDownloader(output_dir=source_vids_dir)



# from youtube_shorts_extractor import YouTubeShortsExtractor

# extractor = YouTubeShortsExtractor(output_dir='data/channels_shorts')
# result = extractor.extract_channel_shorts(
#     channel_url="https://www.youtube.com/@emilfacts",
#     max_videos = None,
#     check_detailed = False,
#     save_to_file = True
#     )

# if result['success']:
#     shorts = result['shorts']  # Sorted by date, newest first
#     print(f"Found {len(shorts)} shorts")
    
#     for short in shorts[:5]:  # First 5 shorts
#         print(f"{short['title']} - {short['webpage_url']}")
        
def dl_batch_vids(source_vids):
    """
    Processes multiple videos using yt_downloader.process_video.
    Each item in source_vids should be a dict with 'url' and 'doc_id'.
    Returns a list of results with doc_id included.
    """
    for vid in source_vids:
        url = vid.get('url')
        vid_id = vid.doc_id # Assuming doc_id is the id of vid in db (used for foler name)
        vid_output_dir = os.path.join(source_vids_dir, str(vid_id).zfill(3))


        res = yt_downloader.process_video(url = url, vid_output_dir=vid_output_dir)
        if res['success'] :
            vid['state'] = cfg['video_state']['downloaded']
            vid['metadata'] = res['metadata']
            vid['source_vid_file_path'] = os.path.join(vid_output_dir, 'source_vid.mp4')
            # update db
            db.update_source_vid_by_id(vid_id, vid)


def dl_batch_vids_already_dled(source_vids):
    """
    Processes multiple videos using yt_downloader.process_video.
    Each item in source_vids should be a dict with 'url' and 'doc_id'.
    Returns a list of results with doc_id included.
    """
    for vid in source_vids:
        url = vid.get('url')
        vid_id = vid.doc_id # Assuming doc_id is the id of vid in db (used for foler name)
        vid_output_dir = os.path.join(source_vids_dir, str(vid_id).zfill(3))

        # get the youtube video id from url
        if 'youtube.com' in url or 'youtu.be' in url:
            if 'shorts' in url:
                yt_video_id = url.split('shorts/')[1].split('?')[0].split('/')[0]
            elif 'watch?v=' in url:
                yt_video_id = url.split('watch?v=')[1].split('&')[0]
            elif 'youtu.be/' in url:
                yt_video_id = url.split('youtu.be/')[1].split('?')[0].split('/')[0]
            else:
                print(f"Unsupported YouTube URL format: {url}")
                continue

        # find the already downloaded video file in the /workspaces/shortgen-try-2/data/source_vids/already_dled_vids folder named [yt_video_id].mp4 and move to vid_output_dir and rename it to source_vid.mp4
        already_dled_vids_dir = os.path.join(source_vids_dir, 'already_dled_vids')
        already_dled_vid_path = os.path.join(already_dled_vids_dir, f"{yt_video_id}.mp4")
        if not os.path.exists(already_dled_vid_path):
            print(f"Already downloaded video file not found: {already_dled_vid_path}")
            continue    
        os.makedirs(vid_output_dir, exist_ok=True)
        target_vid_path = os.path.join(vid_output_dir, 'source_vid.mp4')
        shutil.copy2(already_dled_vid_path, target_vid_path)
        print(f"Copied {already_dled_vid_path} to {target_vid_path}")    

        metadata = {
        "video_info": {
                    "id": yt_video_id,
                    "title": "",
                    "description": "",
                    "thumbnail": f"https://i.ytimg.com/vi/{yt_video_id}/maxresdefault.jpg",
        }
        }

        vid['state'] = cfg['video_state']['downloaded']
        vid['metadata'] = metadata
        vid['source_vid_file_path'] = os.path.join(vid_output_dir, 'source_vid.mp4')

        # download thumbnail
        thumbnail_url = metadata['video_info']['thumbnail']
        thumbnail_path = os.path.join(vid_output_dir, 'source_vid_thumbnail.jpg')
        try:
            response = requests.get(thumbnail_url, timeout=10)
            if response.status_code == 200:
                with open(thumbnail_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download thumbnail: {thumbnail_url} (status {response.status_code})")
        except Exception as e:
            print(f"Error downloading thumbnail: {e}")


        # update db
        db.update_source_vid_by_id(vid_id, vid)