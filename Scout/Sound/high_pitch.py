import numpy as np
import wave

def write_sine_wav(
    filename="tone_18k.wav",
    freq_hz=18000,
    duration_s=3.0,
    sample_rate=48000,
    amplitude=0.2,      # 0.0 to 1.0 (keep low)
    fade_ms=10          # small fade to avoid clicks
):
    # Time vector
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate

    # Sine wave
    x = amplitude * np.sin(2 * np.pi * freq_hz * t)

    # Fade in/out to avoid click
    fade_n = int(sample_rate * (fade_ms / 1000.0))
    if fade_n > 0 and 2 * fade_n < n:
        fade_in = np.linspace(0, 1, fade_n)
        fade_out = np.linspace(1, 0, fade_n)
        x[:fade_n] *= fade_in
        x[-fade_n:] *= fade_out

    # Convert to 16-bit PCM
    pcm = np.int16(np.clip(x, -1.0, 1.0) * 32767)

    # Write WAV (mono)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    print(f"Wrote {filename} | freq={freq_hz} Hz | sr={sample_rate} | dur={duration_s}s")

if __name__ == "__main__":
    write_sine_wav(filename="tone_20k.wav", freq_hz=20000, sample_rate=48000, duration_s=2.0)
