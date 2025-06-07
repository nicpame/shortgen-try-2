# from youtube_downloader import YouTubeDownloader

# # Initialize downloader
# downloader = YouTubeDownloader(output_dir="my_downloads")

# # Process a video
# result = downloader.process_video("https://www.youtube.com/shorts/GCbMNVgRrRI")

# if result['success']:
#     print(f"Downloaded: {result['title']}")
#     print(f"Transcript available: {result['transcript_available']}")
# else:
#     print(f"Error: {result['error']}")


from youtube_shorts_extractor import YouTubeShortsExtractor

extractor = YouTubeShortsExtractor(output_dir='data/channels_shorts')
result = extractor.extract_channel_shorts(
    channel_url="https://www.youtube.com/@emilfacts",
    max_videos = None,
    check_detailed = False,
    save_to_file = True
    )

if result['success']:
    shorts = result['shorts']  # Sorted by date, newest first
    print(f"Found {len(shorts)} shorts")
    
    for short in shorts[:5]:  # First 5 shorts
        print(f"{short['title']} - {short['webpage_url']}")