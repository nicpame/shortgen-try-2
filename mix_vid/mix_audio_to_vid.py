import sys, os

sys.path.append(os.getcwd())

from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips

# config
audio_file_name = "adjusted_speech.wav"
source_video_name = "source_vid.mp4"
output_video_name = "output_video.mp4"


def mix_wav_to_mp4(
    input_video_path: str,
    input_audio_path: str,
    output_video_path: str,
    audio_bitrate: str = "192k",
    video_codec: str = "libx264",
    audio_codec: str = "aac",
):
    """
    Replace the audio track of an MP4 with a WAV file.

    Args:
        input_video_path:  Path to source MP4.
        input_audio_path:  Path to source WAV.
        output_video_path: Path to write the resulting MP4.
        loop_audio:        If True, loop the WAV to match the video duration;
                           otherwise, truncate it.
        audio_bitrate:     Bitrate for the output audio (e.g. "192k").
        video_codec:       Codec for the output video (default H.264).
        audio_codec:       Codec for the output audio (default AAC).
    """
    # Load clips
    video = VideoFileClip(input_video_path)
    audio = AudioFileClip(input_audio_path)

    audio.with_duration(video.duration)

    # Set the new audio in the video
    final = video.with_audio(audio)

    # Write the result to file
    final.write_videofile(
        output_video_path,
        # codec=video_codec,
        # audio_codec=audio_codec,
        # audio_bitrate=audio_bitrate,
        fps=video.fps,
        threads=4,
    )

    # Clean up
    video.close()
    audio.close()
    final.close()


# batch mix audio to video
def batch_mix_audio_to_video(videos_dir: str):
    print(f"Batch mixing audio to video in: {videos_dir}")
    for vid_dir_name in os.listdir(videos_dir):  # List items in the directory
        vid_dir = os.path.join(videos_dir, vid_dir_name)

        # Skip if audio file does not exist in vid_dir
        audio_file_path = os.path.join(vid_dir, audio_file_name)
        if not os.path.isfile(audio_file_path):
            print(f"Audio file does not exist for {vid_dir}, skipping.")
            continue

        if os.path.isdir(vid_dir):  # Check if it's a directory
            video_file_path = os.path.join(
                vid_dir, source_video_name
            )  # Assuming video file is named 'video.mp4'
            output_video_path = os.path.join(vid_dir, output_video_name)

            if os.path.isfile(video_file_path):  # Check if the video file exists
                print(f"Mixing audio to video: {vid_dir}")
                mix_wav_to_mp4(
                    input_video_path=video_file_path,
                    input_audio_path=audio_file_path,
                    output_video_path=output_video_path,
                )
            else:
                print(f"Video file does not exist for {vid_dir}, skipping.")
