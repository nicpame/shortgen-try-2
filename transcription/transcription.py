from groq_transcription import gen_transcription_and_write
from config.config import config


def batch_gen_transcription(root_path):
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