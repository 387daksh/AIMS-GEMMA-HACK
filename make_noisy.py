import torchaudio

# Load both spoken clips
wav1, sr1 = torchaudio.load("voice1.wav")
wav2, sr2 = torchaudio.load("voice2.wav")

# Trim both to the same length (the shorter of the two)
min_len = min(wav1.shape[1], wav2.shape[1])
wav1 = wav1[:, :min_len]
wav2 = wav2[:, :min_len]

# Add the two waveforms together — this is literally what "2 people
# talking at once into 1 mic" sounds like
mixed = wav1 + wav2

torchaudio.save("noisy.wav", mixed, sr1)
print("✅ noisy.wav created — go play it, it should sound like 2 people talking over each other.")