import numpy as np
from scipy.io.wavfile import write

# =========================
# Ambulance Siren Settings
# =========================
sample_rate = 44100
amplitude = 0.8

# Frequencies (pleasant but strong)
f_low = 600
f_high = 1500

# Durations
wail_up = 1.2
wail_down = 1.2
yelp_period = 0.25
yelp_cycles = 12

repeat_blocks = 4   # total ambulance pattern repeats


def sweep(f1, f2, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freqs = f1 + (f2 - f1) * (t / duration)
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
    return np.sin(phase)


def fade(signal, fade_time=0.03):
    fade_len = int(sample_rate * fade_time)
    env = np.ones_like(signal)
    env[:fade_len] = np.linspace(0, 1, fade_len)
    env[-fade_len:] = np.linspace(1, 0, fade_len)
    return signal * env


def generate_wail():
    up = sweep(f_low, f_high, wail_up)
    down = sweep(f_high, f_low, wail_down)
    return fade(np.concatenate((up, down)))


def generate_yelp():
    tones = []
    for i in range(yelp_cycles):
        freq = f_high if i % 2 == 0 else f_low
        t = np.linspace(0, yelp_period, int(sample_rate * yelp_period), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        tones.append(fade(tone, 0.01))
    return np.concatenate(tones)


def main():
    pattern = []

    for _ in range(repeat_blocks):
        pattern.append(generate_wail())
        pattern.append(np.zeros(int(sample_rate * 0.15)))
        pattern.append(generate_yelp())
        pattern.append(np.zeros(int(sample_rate * 0.3)))

    siren = np.concatenate(pattern)
    siren = siren / np.max(np.abs(siren)) * amplitude

    siren_pcm = np.int16(siren * 32767)
    write("ambulance_siren.wav", sample_rate, siren_pcm)

    print("✔ ambulance_siren.wav generated successfully")


if __name__ == "__main__":
    main()
