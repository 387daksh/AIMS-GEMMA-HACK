import os
import torch
from scipy.io import wavfile
from speechbrain.inference.separation import SepformerSeparation as separator
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# Windows Hackathon Fix: SpeechBrain tries to create symlinks which require Admin on Windows.
# We monkeypatch pathlib to copy the files instead!
import pathlib
import shutil
def _safe_symlink(self, target, target_is_directory=False):
    shutil.copy2(target, self)
pathlib.Path.symlink_to = _safe_symlink

class AudioEngine:
    def __init__(self):
        print("[*] Initializing SepFormer Engine... (Downloading model weights if first run)")
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-wham", 
            savedir="pretrained_models/sepformer-wham"
        )
        print("[+] SepFormer Engine fully loaded and standing by.")
        
        print("[*] Initializing Whisper STT Engine...")
        if WhisperModel:
            self.stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
            print("[+] Whisper STT Engine fully loaded and standing by.")
        else:
            self.stt_model = None
            print("[-] faster-whisper not installed. Transcription disabled.")

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribes the given audio file into text."""
        if not self.stt_model:
            return "[Transcription failed: faster-whisper not installed]"
            
        try:
            segments, info = self.stt_model.transcribe(file_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
            return transcript.strip()
        except Exception as e:
            print(f"[-] Transcription error: {e}")
            return f"[Transcription error: {e}]"

    def isolate_audio(self, input_file_path: str) -> str:
        """
        Takes a noisy .wav file path, runs it through the SepFormer neural network,
        isolates the primary voice track, and saves it cleanly using pure scipy.
        """
        if not os.path.exists(input_file_path):
            return f"Error: Input file '{input_file_path}' not found."

        print(f"[*] Processing file: {input_file_path}")
        
        # Create output directory if it doesn't exist
        output_dir = "processed_audio"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up output file naming
        base_name = os.path.basename(input_file_path)
        name_part, _ = os.path.splitext(base_name)
        final_output_path = os.path.join(output_dir, f"{name_part}_isolated.wav")

        try:
            # Perform source separation
            est_sources = self.model.separate_file(path=input_file_path)
            
            # Extract the raw audio tensor data (batch=0, time, channel=0)
            signal = est_sources[0, :, 0].detach().cpu().numpy()
            
            # Normalize signal to prevent clipping artifacts
            import numpy as np
            if np.max(np.abs(signal)) > 0:
                signal = signal / np.max(np.abs(signal))
            
            # Write out file with pure python scipy at 8000 Hz, bypassing all system codecs
            wavfile.write(final_output_path, 8000, (signal * 32767).astype(np.int16))
            
            print(f"[+] Cleaned audio track saved successfully to: {final_output_path}")
            return final_output_path
            
        except Exception as e:
            error_msg = f"Error saving isolated audio stream: {str(e)}"
            print(f"[-] {error_msg}")
            return error_msg

if __name__ == "__main__":
    engine = AudioEngine()
    test_file = "sample_noise.wav"
    
    if os.path.exists(test_file):
        result = engine.isolate_audio(test_file)
        print(f"Test Result Vector: {result}")
    else:
        print(f"[-] Test file '{test_file}' not found. Drop a dummy .wav file in this directory to run the test harness.")