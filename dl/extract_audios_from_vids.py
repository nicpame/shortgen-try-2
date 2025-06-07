import os
from moviepy import VideoFileClip

def process_video(video_path):
    """Extracts audio from a video file and saves it in the same directory with the same name"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio

        if audio:
            audio_filename = os.path.splitext(video_path)[0] + ".mp3"  # Save as MP3 format
            audio.write_audiofile(audio_filename, bitrate="120k") 
            print(f"Audio saved: {audio_filename}")
        
        video.close()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")


def scan_and_process_videos(root_path):
    """Scans the given path for subfolders containing .mp4 files but no audio files"""
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        if os.path.isdir(folder_path):  # Check if it's a subfolder
            video_file = None
            has_audio = False

            for file in os.listdir(folder_path):
                if file.endswith(".mp4"):
                    video_file = os.path.join(folder_path, file)
                elif file.endswith((".mp3", ".wav", ".aac", ".flac")):
                    has_audio = True
            
            if video_file and not has_audio:
                process_video(video_file)


VIDEOS_DIR = "./data/dled_shorts"

scan_and_process_videos(VIDEOS_DIR)
