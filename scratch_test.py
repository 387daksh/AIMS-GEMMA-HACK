import av
import numpy as np
from scipy.io import wavfile
from audio_engine import AudioEngine

def convert_mp3_to_wav(mp3_path, wav_path):
    print(f"[*] Converting {mp3_path} to WAV using PyAV...")
    container = av.open(mp3_path)
    audio_stream = next(s for s in container.streams if s.type == 'audio')
    
    samples = []
    for frame in container.decode(audio_stream):
        samples.append(frame.to_ndarray())
        
    audio_data = np.concatenate(samples, axis=1)
    
    # Check if stereo, convert to mono if necessary
    if audio_data.shape[0] > 1:
        audio_data = np.mean(audio_data, axis=0)
    else:
        audio_data = audio_data[0]
        
    # Ensure it's 16-bit PCM
    if audio_data.dtype != np.int16:
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
    wavfile.write(wav_path, audio_stream.sample_rate, audio_data.T)
    return wav_path

def test_audio():
    mp3_file = "WhatsApp Ptt 2026-07-15 at 1.32.44 PM.mp3"
    wav_file = "test_audio.wav"
    
    convert_mp3_to_wav(mp3_file, wav_file)
    
    print("[*] Initializing engine...")
    engine = AudioEngine()
    
    print(f"[*] Processing file: {wav_file}")
    
    import time
    start_time = time.time()
    res = engine.isolate_audio(wav_file)
    elapsed = time.time() - start_time
    
    print("\n=== RESULTS ===")
    print(f"Result tuple: {res}")
    print(f"Time Taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    test_audio()
