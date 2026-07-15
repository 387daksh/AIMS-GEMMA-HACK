from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

def add_audio_to_video():
    video_path = "Fall2_Cam4.mp4"
    audio_path = "WhatsApp Ptt 2026-07-15 at 1.32.44 PM.mp3"
    output_path = "Fall2_Cam4_with_audio.mp4"
    fall_time = 98.83

    print("[*] Loading video...")
    video = VideoFileClip(video_path)
    
    print("[*] Loading audio...")
    new_audio = AudioFileClip(audio_path).with_start(fall_time)
    
    # If the video already has audio, we mix them. Otherwise we just set the audio.
    if video.audio is not None:
        final_audio = CompositeAudioClip([video.audio, new_audio])
    else:
        final_audio = CompositeAudioClip([new_audio])
        
    final_video = video.with_audio(final_audio)
    
    print(f"[*] Rendering new video to {output_path}...")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    add_audio_to_video()
