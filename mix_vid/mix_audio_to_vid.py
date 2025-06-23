import sys, os
sys.path.append(os.getcwd())

from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips

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

if __name__ == "__main__":
    mix_wav_to_mp4(
        input_video_path="/workspaces/shortgen-try-2/data/dled_shorts/GCbMNVgRrRI/source_video.mp4",
        input_audio_path="/workspaces/shortgen-try-2/data/dled_shorts/GCbMNVgRrRI/output.wav",
        output_video_path="output.mp4",
    )