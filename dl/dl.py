import sys, os
sys.path.append(os.getcwd())
from config.config import config
cfg = config()
from dl.youtube_downloader import YouTubeDownloader

# config
SOURCE_VIDS_DIR = cfg['source_vids_rel_dir']

yt_downloader = YouTubeDownloader(output_dir=SOURCE_VIDS_DIR)



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
    results = []
    for vid in source_vids:
        url = vid.get('url')
        vid_id = vid.doc_id # Assuming doc_id is the id of vid in db (used for foler name)
        vid_output_dir = os.path.join(SOURCE_VIDS_DIR, str(vid_id).zfill(3))
        result = yt_downloader.process_video(url = url, vid_output_dir=vid_output_dir)
        result['source_vid_dir'] = vid_output_dir
        results.append(result)
    return results