import librosa
import soundfile as sf
from audiostretchy.stretch import stretch_audio
import time, os
import json

# config
GENED_SPEECH_FILE_NAME = "gened_speech.wav"
ADJUSTED_SPEECH_FILE_NAME = "adjusted_speech.wav"
METADATA_FILE_NAME = "source_vid_metadata.json"


def adjust_audio_duration(input_path: str, output_path: str, target_duration: float):
    # Initialize processor
    # Load and process: set desired duration directly
    y, sr = librosa.load(input_path, sr=None)
    current_duration = len(y) / sr
    stretch_factor = target_duration / current_duration
    stretch_audio(input_path, output_path, stretch_factor)

    time.sleep(0.5)  # wait to make sure the file is written

    # trim the stretched audio to the target duration coz stretch_audio wont do so
    y, sr = librosa.load(output_path, sr=None)
    start_sample = int(0 * sr)
    end_sample = int(target_duration * sr)
    y_trimmed = y[start_sample:end_sample]
    sf.write(output_path, y_trimmed, sr)


def batch_adjust_audio_duration(videos_dir: str):
    print(f"Batch adjusting audio duration for videos in: {videos_dir}")
    for vid_dir_name in os.listdir(videos_dir):  # List items in the directory
        vid_dir = os.path.join(videos_dir, vid_dir_name)

        if os.path.isdir(vid_dir):  # Check if it's a directory
            audio_file_path = os.path.join(vid_dir, GENED_SPEECH_FILE_NAME)
            if os.path.isfile(audio_file_path):  # Check if the file exists
                # CALCULATE TARGET DURATION IN SECONDS
                target_duration = (
                    get_video_duration_from_metadata(vid_dir) - 0.5
                )  # subtract 0.5 seconds for safety margin

                print(
                    f"Adjusting audio duration for: {vid_dir} to {target_duration} seconds"
                )

                output_path = os.path.join(vid_dir, ADJUSTED_SPEECH_FILE_NAME)
                adjust_audio_duration(audio_file_path, output_path, target_duration)


def get_video_duration_from_metadata(vid_dir: str) -> float:
    metadata_path = os.path.join(vid_dir, METADATA_FILE_NAME)
    with open(metadata_path, "r") as f:
        vid = json.load(f)
    return vid["video_info"]["duration"]
