"""
audio_engine.py
---------------
This is the "audio cleaning tool" for SENTINEL 2.0.

Job: take a noisy .wav clip (e.g. a scream buried under crowd noise) and pull
out the clearest / loudest voice from it, so Gemma 4 gets a cleaner signal.

We use SepFormer, a pretrained model from the SpeechBrain library that's
already trained to separate overlapping speakers. We are NOT training
anything ourselves — just loading weights and running inference.
"""

import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator

print("🔊 Loading SepFormer model... (first run downloads ~150MB, be patient)")

# This checkpoint is trained on WSJ0-2Mix (2 overlapping speakers).
# It's the standard starting point for source separation demos.
model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix",
)


def clean_audio(input_wav_path: str, output_wav_path: str) -> str:
    """
    input_wav_path  -> path to the noisy .wav file (e.g. from Daksh's script)
    output_wav_path -> where we save the cleaned .wav file

    Returns the path to the cleaned file, so the caller (FastAPI) can send it back.
    """
    # est_sources shape: [batch, time, num_speakers] — SepFormer splits the
    # audio into separate "guessed speaker" tracks.
    est_sources = model.separate_file(path=input_wav_path)

    # We don't know in advance which track is the "important" one, so as a
    # simple heuristic we just pick whichever track has the most energy
    # (i.e. is loudest overall). Good enough for a hackathon demo.
    energies = [
        est_sources[:, :, i].abs().mean().item()
        for i in range(est_sources.shape[2])
    ]
    best_speaker_idx = energies.index(max(energies))

    cleaned = est_sources[:, :, best_speaker_idx].detach().cpu()

    # SepFormer's pretrained models work at 8kHz sample rate
    torchaudio.save(output_wav_path, cleaned, 8000)
    return output_wav_path


if __name__ == "__main__":
    # Quick manual test before wiring this into FastAPI:
    # put a file called noisy.wav next to this script, then run:
    #     python audio_engine.py
    out = clean_audio("noisy.wav", "cleaned.wav")
    print(f"✅ Done! Cleaned audio saved to: {out}")