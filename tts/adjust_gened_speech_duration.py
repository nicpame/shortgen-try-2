import librosa
import soundfile as sf
import numpy as np

def change_playback_speed_to_duration(input_file, output_file, target_duration):
    """
    Change the playback speed of a WAV file to match a target duration.
    
    Args:
        input_file (str): Path to input WAV file
        output_file (str): Path to output WAV file
        target_duration (float): Desired duration in seconds
    """
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Calculate current duration
    current_duration = len(y) / sr
    
    # Calculate speed factor needed
    speed_factor = current_duration / target_duration
    
    # Change the speed using librosa's time_stretch
    y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
    
    # Save the result
    sf.write(output_file, y_stretched, sr)
    
    print(f"Original duration: {current_duration:.2f} seconds")
    print(f"Target duration: {target_duration:.2f} seconds")
    print(f"Speed factor: {speed_factor:.2f}x")
    print(f"Saved to: {output_file}")

import librosa
import soundfile as sf
import pyrubberband as pyrb

def change_speed_rubberband(input_file, output_file, target_duration):
    """
    Professional-quality time stretching using Rubber Band
    """
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    current_duration = len(y) / sr
    stretch_factor = target_duration / current_duration
    
    # Use Rubber Band for high-quality stretching
    y_stretched = pyrb.time_stretch(y, sr, stretch_factor)
    
    sf.write(output_file, y_stretched, sr)
    print(f"High-quality stretch: {current_duration:.2f}s → {target_duration:.2f}s")


test = change_speed_rubberband
test('data/dled_shorts/GCbMNVgRrRI/gened_speech.wav', 'output.wav', 60.0) 