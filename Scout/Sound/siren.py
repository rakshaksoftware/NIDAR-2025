import numpy as np
from scipy.io.wavfile import write

# ------------------------------
# Pleasant Emergency Siren Config
# ------------------------------
sample_rate = 44100          # Hz
f_start = 700                # Hz (lower, warmer)
f_end = 1600                 # Hz (still sharp but not piercing)
duration_up = 0.9            # seconds
duration_down = 0.9          # seconds
silence_between = 0.1        # seconds
repeat_cycles = 6            # how many times to repeat
amplitude = 0.75             # 0.0–1.0 (0.75 is loud but not clipping)


def smooth_sweep(f1, f2, duration, sr, amp):
    """Generate a smooth frequency sweep with fade-in/out."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Linear frequency sweep
    freqs = f1 + (f2 - f1) * (t / duration)

    # Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(freqs) / sr
    sweep = np.sin(phase)

    # Apply fade-in/out (Hann-like envelope)
    env = np.ones_like(sweep)
    fade_len = int(0.03 * sr)  # 30 ms fade
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        env[:fade_len] *= fade_in
        env[-fade_len:] *= fade_out

    return amp * sweep * env


def generate_pleasant_siren():
    # One up + one down sweep
    sweep_up = smooth_sweep(f_start, f_end, duration_up, sample_rate, amplitude)
    sweep_down = smooth_sweep(f_end, f_start, duration_down, sample_rate, amplitude)

    # Small silence between cycles
    silence = np.zeros(int(sample_rate * silence_between))

    one_cycle = np.concatenate((sweep_up, sweep_down, silence))

    # Repeat the pattern
    siren = np.tile(one_cycle, repeat_cycles)

    # Normalize just in case
    max_val = np.max(np.abs(siren))
    if max_val > 0:
        siren = siren / max_val * 0.95  # keep headroom

    # Convert to 16-bit PCM and save
    siren_pcm = np.int16(siren * 32767)
    write("siren_pleasant.wav", sample_rate, siren_pcm)
    print("✔ Generated siren_pleasant.wav")


if __name__ == "__main__":
    generate_pleasant_siren()
