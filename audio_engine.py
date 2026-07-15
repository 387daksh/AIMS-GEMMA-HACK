import os
import torch
from scipy.io import wavfile
# --- MONKEY PATCH SPEECHBRAIN LAZY MODULE INSPECT BUG ---
# SpeechBrain's LazyModule crashes inspect.getmodule on Windows during torchaudio load
try:
    import speechbrain.utils.importutils
    _orig_getattr = speechbrain.utils.importutils.LazyModule.__getattr__
    def _safe_getattr(self, attr):
        if attr == "__file__" and not self.loaded:
            raise AttributeError("LazyModule has no __file__ yet")
        return _orig_getattr(self, attr)
    speechbrain.utils.importutils.LazyModule.__getattr__ = _safe_getattr
except Exception:
    pass
# --------------------------------------------------------

from speechbrain.inference.separation import SepformerSeparation as separator
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# Windows Hackathon Fix: SpeechBrain tries to create symlinks which require Admin on Windows.
# We monkeypatch pathlib to copy the files instead! (Windows-only hack)
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
            
            # Extract both channels
            signal_0 = est_sources[0, :, 0].detach().cpu().numpy()
            signal_1 = est_sources[0, :, 1].detach().cpu().numpy()
            
            # Save both to temporary files
            temp_0 = "processed_audio/channel_0_temp.wav"
            temp_1 = "processed_audio/channel_1_temp.wav"
            import numpy as np
            wavfile.write(temp_0, 8000, (signal_0 * 32767).astype(np.int16))
            wavfile.write(temp_1, 8000, (signal_1 * 32767).astype(np.int16))
            
            # Score each channel by transcribing
            text_0 = self.transcribe_audio(temp_0)
            text_1 = self.transcribe_audio(temp_1)
            
            # Pick the channel with the longest text (most words), or fallback to channel 0
            if len(text_1) > len(text_0):
                best_signal = signal_1
                transcript = text_1
            else:
                best_signal = signal_0
                transcript = text_0
                
            os.remove(temp_0)
            os.remove(temp_1)

            # Save the best clean signal
            out_name = f"isolated_{base_name}"
            out_path = os.path.join("processed_audio", out_name)
            wavfile.write(out_path, 8000, (best_signal * 32767).astype(np.int16))
            
            print(f"[+] Cleaned audio track saved successfully to: {out_path}")
            return out_path, transcript
        except Exception as e:
            import traceback
            error_msg = f"Error saving isolated audio stream: {str(e)}\n{traceback.format_exc()}"
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