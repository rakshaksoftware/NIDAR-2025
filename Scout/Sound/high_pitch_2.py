import numpy as np
import wave

def _normalize_to_peak(x, peak=0.99):
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)

def _fade_edges(x, sr, fade_ms=8):
    n = len(x)
    f = int(sr * fade_ms / 1000)
    if f <= 0 or 2*f >= n:
        return x
    ramp = np.linspace(0, 1, f)
    x[:f] *= ramp
    x[-f:] *= ramp[::-1]
    return x

def _rich_tone(t, f, style="sine", harmonics=3):
    """
    style:
      - "sine"   : pure
      - "rich"   : sine + harmonics (cuts through better)
      - "square" : approximated by odd harmonics (very attention-grabbing)
    """
    if style == "sine":
        return np.sin(2*np.pi*f*t)

    if style == "rich":
        y = np.zeros_like(t)
        for k in range(1, harmonics+1):
            y += (1.0/k) * np.sin(2*np.pi*(k*f)*t)
        return y

    if style == "square":
        y = np.zeros_like(t)
        # odd harmonics only
        for k in range(1, 2*harmonics, 2):
            y += (1.0/k) * np.sin(2*np.pi*(k*f)*t)
        return y

    raise ValueError("Unknown style")

def generate_drone_alert_wav(
    filename="DRONE_ALERT.wav",
    sr=48000,
    total_s=12.0,

    # Frequencies chosen to be very audible over prop noise
    f1=1000,        # Hz
    f2=1600,        # Hz

    # Warble inside each "ON" burst
    warble_hz=8.0,  # toggles f1/f2 ~8 times/sec

    # Cadence (on/off) helps detection in chaos
    on_s=0.55,
    off_s=0.20,

    # Tone character (square is most piercing, rich is good compromise)
    style="square",     # "sine" / "rich" / "square"
    harmonics=5,

    # Safety: keep <1.0; we will normalize near full-scale
    peak=0.99
):
    n_total = int(sr * total_s)
    y = np.zeros(n_total, dtype=np.float32)

    block_len = int(sr * (on_s + off_s))
    on_len = int(sr * on_s)

    # Build repeating ON/OFF blocks
    idx = 0
    while idx < n_total:
        # ON segment
        seg_len = min(on_len, n_total - idx)
        t = np.arange(seg_len) / sr

        # warble: switch between f1 and f2 using a square LFO
        lfo = np.sign(np.sin(2*np.pi*warble_hz*t))  # +/-1
        f = np.where(lfo >= 0, f1, f2)

        # generate tone with harmonics
        seg = np.zeros_like(t)
        # vectorized-ish: accumulate harmonics by loops (fast enough)
        if style in ("rich", "square"):
            seg = np.zeros_like(t)
            if style == "rich":
                ks = range(1, harmonics+1)
            else:  # square
                ks = range(1, 2*harmonics, 2)

            for k in ks:
                seg += (1.0/k) * np.sin(2*np.pi*(k*f)*t)

        else:
            seg = np.sin(2*np.pi*f*t)

        seg = _fade_edges(seg, sr, fade_ms=8)

        y[idx:idx+seg_len] += seg.astype(np.float32)

        # advance by whole block
        idx += block_len

    # Normalize to near full-scale
    y = _normalize_to_peak(y, peak=peak)

    # Convert to 16-bit PCM mono WAV
    pcm = np.int16(np.clip(y, -1.0, 1.0) * 32767)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    print(f"Wrote {filename} | {total_s}s | {f1}/{f2}Hz warble @ {warble_hz}Hz | style={style}")

if __name__ == "__main__":
    generate_drone_alert_wav()
