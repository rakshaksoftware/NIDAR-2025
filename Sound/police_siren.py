import numpy as np
from scipy.io.wavfile import write

# ======================
# Siren configuration
# ======================
sample_rate = 44100   # Hz
duration = 5.0        # seconds
low_freq = 700        # Hz (low tone)
high_freq = 1200      # Hz (high tone)
sweep_rate = 0.6      # Hz (how fast it goes hi-lo)

# Time array
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Smooth frequency sweep (sinusoidal modulation)
inst_freq = low_freq + (high_freq - low_freq) * (0.5 * (1 + np.sin(2 * np.pi * sweep_rate * t)))

# Generate siren waveform
phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
siren = np.sin(phase)

# Normalize to prevent clipping
siren /= np.max(np.abs(siren))

# Save WAV file
write("police_siren.wav", sample_rate, siren.astype(np.float32))

print("police_siren.wav generated successfully")
