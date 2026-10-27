"""Shared audio loading, speech download, analysis, and plots."""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

_CACHE = Path.home() / ".cache" / "audio-lab"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("LIBROSA_DATA_DIR", str(_CACHE / "librosa"))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import requests
import soundfile as sf
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scipy import signal as sp_signal

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "downloads"
DOWNLOADS.mkdir(exist_ok=True)

CMAP = "magma"

VOICES = [
    ("한국어 · 여자", "ko-KR-SunHiNeural"),
    ("한국어 · 남자", "ko-KR-InJoonNeural"),
    ("English · Woman", "en-US-JennyNeural"),
    ("English · Man", "en-US-GuyNeural"),
]

SAMPLES = [
    ("LibriSpeech 1 · 영어 낭독", "libri1"),
    ("LibriSpeech 2 · 영어 낭독", "libri2"),
    ("LibriSpeech 3 · 영어 낭독", "libri3"),
]

DEMOS = [
    ("순음 440 Hz (A4)", "sine_440"),
    ("로그 Chirp 200→4000 Hz", "chirp"),
    ("두 사인파 440 + 880 Hz", "two_tone"),
    ("백색 잡음", "noise"),
    ("진폭 변조 (AM)", "am"),
]

DEFAULT_TTS_TEXT = "안녕하세요. 이 문장을 음성으로 받아 재생하고, 파형과 멜 스펙트로그램을 분석합니다."

plt.rcParams.update(
    {
        "font.family": ["AppleGothic", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#0f1117",
        "axes.edgecolor": "#3d4454",
        "axes.labelcolor": "#d7dce6",
        "xtick.color": "#9aa3b5",
        "ytick.color": "#9aa3b5",
        "text.color": "#e8ecf4",
        "savefig.facecolor": "#0f1117",
        "savefig.edgecolor": "#0f1117",
    }
)


@dataclass
class AudioClip:
    y: np.ndarray
    sr: int
    name: str
    channels: int
    path: str | None = None

    @property
    def duration(self) -> float:
        return float(len(self.y) / self.sr) if self.sr else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.y))) if len(self.y) else 0.0


def _stamp(prefix: str, suffix: str) -> Path:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DOWNLOADS / f"{prefix}_{now}{suffix}"


def _safe_stem(text: str, limit: int = 24) -> str:
    cleaned = re.sub(r"[^\w가-힣]+", "_", text).strip("_")
    return (cleaned or "speech")[:limit]


def load_audio_file(path: str | Path) -> AudioClip:
    path = Path(path)
    y, sr = librosa.load(path, sr=None, mono=True)
    try:
        channels = int(sf.info(path).channels)
    except Exception:
        channels = 1
    return AudioClip(y=y, sr=sr, name=path.name, channels=channels, path=str(path))


def load_audio_bytes(raw: bytes, name: str) -> AudioClip:
    y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
    try:
        channels = int(sf.info(io.BytesIO(raw)).channels)
    except Exception:
        channels = 1
    return AudioClip(y=y, sr=sr, name=name, channels=channels)


def save_clip_wav(clip: AudioClip, path: Path | None = None) -> Path:
    path = path or _stamp("clip", ".wav")
    sf.write(path, clip.y, clip.sr, format="WAV")
    return path


def make_demo(kind: str, sr: int = 22050, seconds: float = 3.0) -> AudioClip:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    names = {
        "sine_440": "demo_sine_440hz.wav",
        "chirp": "demo_chirp.wav",
        "two_tone": "demo_two_tone.wav",
        "noise": "demo_white_noise.wav",
        "am": "demo_am.wav",
    }
    if kind == "sine_440":
        y = 0.6 * np.sin(2 * np.pi * 440 * t)
    elif kind == "chirp":
        y = 0.55 * sp_signal.chirp(t, f0=200, f1=4000, t1=seconds, method="logarithmic")
    elif kind == "two_tone":
        y = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t)
    elif kind == "noise":
        y = 0.25 * np.random.default_rng(7).standard_normal(t.shape)
    else:
        y = 0.6 * np.sin(2 * np.pi * 600 * t) * (0.5 * (1 + np.sin(2 * np.pi * 4 * t)))
    fade = int(0.02 * sr)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    clip = AudioClip(y=y.astype(np.float32), sr=sr, name=names.get(kind, "demo.wav"), channels=1)
    path = save_clip_wav(clip, DOWNLOADS / clip.name)
    clip.path = str(path)
    return clip


def synthesize_speech(text: str, voice: str = "ko-KR-SunHiNeural") -> Path:
    import asyncio
    import edge_tts

    text = text.strip()
    if not text:
        raise ValueError("변환할 문장을 입력하세요.")

    dest = _stamp(f"tts_{_safe_stem(text)}_{voice.split('-')[0]}", ".mp3")

    async def _run() -> None:
        await edge_tts.Communicate(text, voice).save(str(dest))

    asyncio.run(_run())
    return dest


def download_url(url: str) -> Path:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("http(s) URL을 입력하세요.")
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
        suffix = ".wav"
    dest = _stamp("url", suffix)
    response = requests.get(url, timeout=30, headers={"User-Agent": "AudioLab/1.0"})
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def download_sample(key: str) -> Path:
    src = Path(librosa.ex(key))
    dest = DOWNLOADS / f"sample_{key}{src.suffix}"
    if not dest.exists():
        dest.write_bytes(src.read_bytes())
    return dest


def list_downloads() -> list[Path]:
    files = [p for p in DOWNLOADS.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def analyze(clip: AudioClip, hop: int = 512) -> dict[str, float]:
    rms = float(np.mean(librosa.feature.rms(y=clip.y, hop_length=hop)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(clip.y, hop_length=hop)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=clip.y, sr=clip.sr, hop_length=hop)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=clip.y, sr=clip.sr, hop_length=hop)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=clip.y, sr=clip.sr, hop_length=hop)))
    f0 = librosa.yin(clip.y, fmin=65, fmax=400, sr=clip.sr, hop_length=hop)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "f0": float(np.median(voiced)) if len(voiced) else 0.0,
    }


def _style_ax(ax: Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=8, color="#e8ecf4")
    ax.grid(True, alpha=0.18)


def draw_waveform(ax: Axes, clip: AudioClip, playhead: float | None = None) -> None:
    ax.clear()
    times = np.arange(len(clip.y)) / clip.sr
    ax.plot(times, clip.y, color="#7aa2ff", linewidth=0.6)
    if playhead is not None:
        ax.axvline(playhead, color="#ff6b8a", linewidth=1.1)
    ax.set_xlim(0, max(clip.duration, 1e-3))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    _style_ax(ax, "Waveform")


def draw_mel(ax: Axes, clip: AudioClip, n_fft: int, hop: int, n_mels: int, fmin: int, fmax: int):
    ax.clear()
    fmax = min(fmax, clip.sr // 2)
    S = librosa.feature.melspectrogram(
        y=clip.y, sr=clip.sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_db, sr=clip.sr, hop_length=hop, x_axis="time", y_axis="mel", fmin=fmin, fmax=fmax, cmap=CMAP, ax=ax
    )
    _style_ax(ax, "Mel Spectrogram (dB)")
    return img


def draw_stft(ax: Axes, clip: AudioClip, n_fft: int, hop: int):
    ax.clear()
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(clip.y, n_fft=n_fft, hop_length=hop)), ref=np.max)
    img = librosa.display.specshow(S_db, sr=clip.sr, hop_length=hop, x_axis="time", y_axis="hz", cmap=CMAP, ax=ax)
    _style_ax(ax, "Linear Spectrogram (STFT, dB)")
    return img


def draw_spectrum(ax: Axes, clip: AudioClip) -> None:
    ax.clear()
    n = len(clip.y)
    mag = np.abs(np.fft.rfft(clip.y * np.hanning(n)))
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    mag_db -= np.max(mag_db)
    ax.plot(np.fft.rfftfreq(n, d=1 / clip.sr), mag_db, color="#ffb86b", linewidth=0.8)
    ax.set_xlim(0, clip.sr / 2)
    ax.set_ylim(-80, 3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    _style_ax(ax, "Average Spectrum")


def draw_pitch(ax: Axes, clip: AudioClip, hop: int) -> None:
    ax.clear()
    f0 = librosa.yin(clip.y, fmin=65, fmax=400, sr=clip.sr, hop_length=hop)
    times = librosa.times_like(f0, sr=clip.sr, hop_length=hop)
    f0 = np.where(np.isfinite(f0), f0, np.nan)
    ax.plot(times, f0, color="#f0a8ff", linewidth=1.2)
    ax.set_xlim(0, max(clip.duration, 1e-3))
    ax.set_ylim(60, 420)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    _style_ax(ax, "Pitch (YIN)")


def draw_features(ax_or_axes, clip: AudioClip, hop: int) -> None:
    axes = ax_or_axes if isinstance(ax_or_axes, np.ndarray) else [ax_or_axes]
    rms = librosa.feature.rms(y=clip.y, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(clip.y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=clip.y, sr=clip.sr, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=clip.sr, hop_length=hop)
    series = [(rms, "#7ee787", "RMS"), (zcr, "#79c0ff", "ZCR"), (centroid, "#ffa657", "Centroid (Hz)")]
    if len(axes) == 1:
        ax = axes[0]
        ax.clear()
        ax.plot(times, rms, color="#7ee787", label="RMS")
        ax.plot(times, zcr * np.max(rms) / max(np.max(zcr), 1e-9), color="#79c0ff", label="ZCR (scaled)")
        ax.legend(facecolor="#161a22", edgecolor="#2a3140", labelcolor="#d7dce6")
        _style_ax(ax, "Frame-level Features")
        return
    for ax, (data, color, label) in zip(axes, series):
        ax.clear()
        ax.plot(times, data, color=color)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.18)
    axes[0].set_title("Frame-level Features", loc="left", pad=8, color="#e8ecf4")
    axes[-1].set_xlabel("Time (s)")


def _new_fig(size: tuple[float, float]) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=size, dpi=140)
    return fig, ax


def plot_waveform(clip: AudioClip) -> Figure:
    fig, ax = _new_fig((12, 2.6))
    draw_waveform(ax, clip)
    fig.tight_layout()
    return fig


def plot_mel(clip: AudioClip, n_fft: int, hop: int, n_mels: int, fmin: int, fmax: int) -> Figure:
    fig, ax = _new_fig((12, 4.0))
    img = draw_mel(ax, clip, n_fft, hop, n_mels, fmin, fmax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    fig.tight_layout()
    return fig


def plot_stft(clip: AudioClip, n_fft: int, hop: int) -> Figure:
    fig, ax = _new_fig((12, 4.0))
    img = draw_stft(ax, clip, n_fft, hop)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    fig.tight_layout()
    return fig


def plot_spectrum(clip: AudioClip) -> Figure:
    fig, ax = _new_fig((12, 3.2))
    draw_spectrum(ax, clip)
    fig.tight_layout()
    return fig


def plot_features(clip: AudioClip, hop: int) -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.2), dpi=140, sharex=True)
    draw_features(axes, clip, hop)
    fig.tight_layout()
    return fig


def plot_pitch(clip: AudioClip, hop: int) -> Figure:
    fig, ax = _new_fig((12, 3.2))
    draw_pitch(ax, clip, hop)
    fig.tight_layout()
    return fig


def plot_mfcc(clip: AudioClip, n_fft: int, hop: int, n_mels: int) -> Figure:
    mfcc = librosa.feature.mfcc(y=clip.y, sr=clip.sr, n_mfcc=13, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    fig, ax = _new_fig((12, 3.6))
    vmax = float(np.max(np.abs(mfcc))) or 1.0
    img = librosa.display.specshow(
        mfcc, sr=clip.sr, hop_length=hop, x_axis="time", cmap="coolwarm", norm=Normalize(vmin=-vmax, vmax=vmax), ax=ax
    )
    ax.set_ylabel("MFCC")
    _style_ax(ax, "MFCC (13 coefficients)")
    fig.colorbar(img, ax=ax, pad=0.02)
    fig.tight_layout()
    return fig


def resample_clip(clip: AudioClip, target_sr: int) -> AudioClip:
    if target_sr == clip.sr:
        y = clip.y.copy()
    else:
        y = librosa.resample(clip.y, orig_sr=clip.sr, target_sr=target_sr)
    stem = Path(clip.name).stem
    return AudioClip(y=y.astype(np.float32), sr=target_sr, name=f"{stem}_{target_sr}hz.wav", channels=1)


@dataclass
class Segment:
    start: float
    end: float
    label: str
    code: int


_LABEL_KO = {0: "배경(background)", 1: "무성음(unvoiced)", 2: "유성음(voiced)"}


def _frames_to_segments(times: np.ndarray, labels: np.ndarray, hop_s: float, min_dur: float) -> list[Segment]:
    if len(labels) == 0:
        return []
    raw: list[Segment] = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            start_t = max(0.0, times[start_idx] - hop_s / 2)
            end_t = times[i - 1] + hop_s / 2
            raw.append(Segment(start=start_t, end=end_t, label=_LABEL_KO[labels[start_idx]], code=int(labels[start_idx])))
            start_idx = i
    merged: list[Segment] = []
    for seg in raw:
        if merged and (seg.end - seg.start) < min_dur:
            prev = merged[-1]
            merged[-1] = Segment(start=prev.start, end=seg.end, label=prev.label, code=prev.code)
        else:
            merged.append(seg)
    # Swallowing a short blip above can leave two same-label segments
    # adjacent to each other; coalesce those into one.
    coalesced: list[Segment] = []
    for seg in merged:
        if coalesced and coalesced[-1].code == seg.code:
            prev = coalesced[-1]
            coalesced[-1] = Segment(start=prev.start, end=seg.end, label=prev.label, code=prev.code)
        else:
            coalesced.append(seg)
    return coalesced


def classify_segments(
    clip: AudioClip,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_margin_db: float = 12.0,
    zcr_threshold: float = 0.12,
    min_segment_ms: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, list[Segment], dict]:
    """Heuristic background(0)/unvoiced(1)/voiced(2) classifier.

    Background: short-time energy near the estimated noise floor (10th percentile).
    Voiced: energy above the floor, with either a reliable YIN pitch or low ZCR.
    Unvoiced: energy above the floor but high ZCR and no reliable pitch (fricatives/stops).

    Returns (times, labels, segments, diagnostics) — diagnostics carries the
    frame-level RMS(dB)/ZCR curves plus the threshold and framing (frame/hop
    length, overlap %) used, for plotting alongside the classification.
    """
    sr = clip.sr
    frame_length = max(int(sr * frame_ms / 1000), 32)
    hop_length = max(int(sr * hop_ms / 1000), 16)

    rms = librosa.feature.rms(y=clip.y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(clip.y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))
    noise_floor_db = float(np.percentile(rms_db, 10))
    energy_threshold_db = noise_floor_db + energy_margin_db

    f0, voiced_flag, _ = librosa.pyin(
        clip.y, fmin=65, fmax=400, sr=sr, frame_length=frame_length * 4, hop_length=hop_length
    )
    voiced_pitch = voiced_flag[: len(times)] & np.isfinite(f0[: len(times)])

    labels = np.zeros(len(times), dtype=int)
    active = rms_db > energy_threshold_db
    labels[active] = 1
    labels[active & (voiced_pitch | (zcr < zcr_threshold))] = 2

    if len(labels) >= 3:
        labels = sp_signal.medfilt(labels, kernel_size=3).astype(int)

    segments = _frames_to_segments(times, labels, hop_ms / 1000, min_segment_ms / 1000)
    diagnostics = {
        "rms_db": rms_db,
        "zcr": zcr,
        "threshold_db": energy_threshold_db,
        "noise_floor_db": noise_floor_db,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "overlap_pct": max(0.0, (1 - hop_ms / frame_ms) * 100),
    }
    return times, labels, segments, diagnostics


_ASR_MODELS: dict[str, object] = {}


def transcribe(clip: AudioClip, model_size: str = "base") -> dict:
    """Transcribe a clip locally with faster-whisper (no API key, runs on CPU)."""
    from faster_whisper import WhisperModel

    model = _ASR_MODELS.get(model_size)
    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=str(_CACHE / "whisper"))
        _ASR_MODELS[model_size] = model

    y16k = clip.y if clip.sr == 16000 else librosa.resample(clip.y, orig_sr=clip.sr, target_sr=16000)
    raw_segments, info = model.transcribe(y16k.astype(np.float32), beam_size=5)
    segments = [(float(s.start), float(s.end), s.text.strip()) for s in raw_segments]
    return {
        "text": " ".join(s[2] for s in segments).strip(),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segments,
    }


def to_wav_bytes(clip: AudioClip) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, clip.y, clip.sr, format="WAV")
    return buf.getvalue()


def _plotly_theme(title: str, height: int, t0: float, t1: float):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color="#f4f6fb")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10131c",
        font=dict(color="#c5cce0", family="Pretendard, Apple SD Gothic Neo, sans-serif", size=12),
        margin=dict(l=56, r=16, t=48, b=48),
        height=height,
        xaxis=dict(
            title="Time (s)",
            range=[t0, t1],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            rangeslider=dict(visible=True, thickness=0.08, bgcolor="#161a22"),
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        coloraxis_colorbar=dict(title="dB", thickness=12, len=0.8),
    )
    return fig


def plotly_waveform(clip: AudioClip, t0: float, t1: float):
    times = np.arange(len(clip.y)) / clip.sr
    fig = _plotly_theme("Waveform", 260, t0, t1)
    fig.add_scatter(x=times, y=clip.y, mode="lines", line=dict(color="#8eb4ff", width=1), hovertemplate="%{x:.3f}s<br>%{y:.3f}<extra></extra>")
    fig.update_yaxes(title="Amplitude")
    return fig


def _mel_data(clip: AudioClip, n_fft: int, hop: int, n_mels: int, fmin: int, fmax: int):
    fmax = min(fmax, clip.sr // 2)
    S = librosa.feature.melspectrogram(
        y=clip.y, sr=clip.sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    times = librosa.times_like(S_db, sr=clip.sr, hop_length=hop)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    return times, freqs, S_db


def plotly_mel(clip: AudioClip, n_fft: int, hop: int, n_mels: int, fmin: int, fmax: int, t0: float, t1: float, y0: float, y1: float):
    times, freqs, S_db = _mel_data(clip, n_fft, hop, n_mels, fmin, fmax)
    fig = _plotly_theme("Mel Spectrogram", 460, t0, t1)
    fig.add_heatmap(
        z=S_db,
        x=times,
        y=freqs,
        colorscale="Magma",
        colorbar=dict(title="dB", thickness=12),
        hovertemplate="Time %{x:.3f}s<br>Mel %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
    )
    fig.update_yaxes(title="Mel (Hz)", range=[y0, y1])
    return fig


def plotly_stft(clip: AudioClip, n_fft: int, hop: int, t0: float, t1: float, y0: float, y1: float):
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(clip.y, n_fft=n_fft, hop_length=hop)), ref=np.max)
    times = librosa.times_like(S_db, sr=clip.sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=clip.sr, n_fft=n_fft)
    fig = _plotly_theme("Linear Spectrogram (STFT)", 460, t0, t1)
    fig.add_heatmap(
        z=S_db,
        x=times,
        y=freqs,
        colorscale="Magma",
        colorbar=dict(title="dB", thickness=12),
        hovertemplate="Time %{x:.3f}s<br>%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
    )
    fig.update_yaxes(title="Frequency (Hz)", range=[y0, y1])
    return fig


# ---------------------------------------------------------------------------
# HW3: short-time framing / OLA, pitch estimation (autocorrelation & AMDF), VAD
# ---------------------------------------------------------------------------


@dataclass
class FrameSet:
    frames: np.ndarray  # (n_frames, frame_length), windowed
    window: np.ndarray  # (frame_length,)
    frame_length: int
    hop_length: int
    sr: int


def frame_signal(clip: AudioClip, frame_ms: float = 20.0, overlap_pct: float = 50.0, window: str = "hamming") -> FrameSet:
    """Split a clip into short-time, windowed, overlapping frames."""
    sr = clip.sr
    frame_length = max(int(round(sr * frame_ms / 1000)), 2)
    hop_length = max(int(round(frame_length * (1 - overlap_pct / 100))), 1)
    win = sp_signal.get_window(window, frame_length, fftbins=True).astype(np.float32)
    y = clip.y
    n_frames = 1 + max(0, (len(y) - frame_length) // hop_length)
    if n_frames <= 0:
        frames = np.zeros((0, frame_length), dtype=np.float32)
    else:
        frames = np.stack([y[i * hop_length: i * hop_length + frame_length] * win for i in range(n_frames)])
    return FrameSet(frames=frames, window=win, frame_length=frame_length, hop_length=hop_length, sr=sr)


def ola_reconstruct(fs: FrameSet, weighted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Overlap-and-add reconstruction. weighted=True normalizes by the summed
    window envelope (weighted OLA), which compensates the amplitude ripple a
    window that doesn't satisfy the constant-overlap-add (COLA) constraint
    introduces (e.g. Hamming at 50% overlap, unlike Hanning)."""
    n_frames = len(fs.frames)
    output_len = fs.hop_length * (n_frames - 1) + fs.frame_length if n_frames else 0
    y = np.zeros(output_len, dtype=np.float64)
    wsum = np.zeros(output_len, dtype=np.float64)
    for i, frame in enumerate(fs.frames):
        s = i * fs.hop_length
        y[s: s + fs.frame_length] += frame
        wsum[s: s + fs.frame_length] += fs.window
    if weighted:
        safe = np.where(np.abs(wsum) < 1e-6, 1.0, wsum)
        y = y / safe
    return y.astype(np.float32), wsum


def window_overlap_profile(window: str, frame_length: int, hop_length: int, n_frames: int = 10) -> np.ndarray:
    """Sum of shifted copies of a window — visualizes whether it satisfies
    the constant-overlap-add (COLA) property at this hop size."""
    win = sp_signal.get_window(window, frame_length, fftbins=True)
    output_len = hop_length * (n_frames - 1) + frame_length
    wsum = np.zeros(output_len)
    for i in range(n_frames):
        s = i * hop_length
        wsum[s: s + frame_length] += win
    return wsum


def _autocorr_pitch_frame(frame: np.ndarray, sr: int, lag_min: int, lag_max: int) -> tuple[float, float]:
    x = frame - np.mean(frame)
    n = len(x)
    lag_max_eff = min(lag_max, n - 1)
    if lag_min >= lag_max_eff:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2:]
    if ac[0] <= 1e-9:
        return 0.0, 0.0
    lags = np.arange(lag_min, lag_max_eff + 1)
    cross = ac[lag_min: lag_max_eff + 1]
    # Normalize by the geometric mean of the two overlapping segments'
    # energies (a normalized cross-correlation, NCCF) rather than by ac[0]:
    # a raw dot-product's magnitude mechanically shrinks as lag grows (fewer
    # samples overlap), which biases a plain argmax toward short lags/high
    # frequencies regardless of periodicity.
    x2 = x * x
    cum_e = np.concatenate(([0.0], np.cumsum(x2)))
    e_head = cum_e[n - lags]
    e_tail = cum_e[n] - cum_e[lags]
    denom = np.sqrt(e_head * e_tail)
    scores = np.where(denom > 1e-9, cross / denom, 0.0)
    peak = int(np.argmax(scores))
    lag = lag_min + peak
    f0 = sr / lag if lag > 0 else 0.0
    conf = float(max(scores[peak], 0.0))
    if peak == 0 or peak == len(scores) - 1:
        # A peak sitting exactly at the edge of the search range usually
        # means the search boundary won by default (no real periodicity was
        # found in range), not that a genuine period lives right there --
        # down-weight it instead of trusting it as a confident pitch.
        conf *= 0.3
    return f0, conf


def _amdf_pitch_frame(frame: np.ndarray, sr: int, lag_min: int, lag_max: int) -> tuple[float, float]:
    n = len(frame)
    lag_max_eff = min(lag_max, n - 1)
    if lag_min >= lag_max_eff:
        return 0.0, 0.0
    lags = np.arange(lag_min, lag_max_eff + 1)
    amdf = np.array([np.mean(np.abs(frame[: n - lag] - frame[lag:])) for lag in lags])
    orig_dip = int(np.argmin(amdf))
    orig_lag = lag_min + orig_dip
    # AMDF can lock onto an exact integer multiple of the true period -- its
    # dip there can be *deeper* than at the true fundamental, especially when
    # the true period isn't a whole number of samples (a multiple is more
    # likely to land on an integer lag with near-zero residual). Prefer the
    # shortest sub-multiple of the found lag whose dip is comparably deep.
    best_lag = orig_lag
    for k in (2, 3, 4, 5):
        cand_lag = orig_lag // k
        if cand_lag < lag_min:
            break
        cand_idx = cand_lag - lag_min
        if amdf[cand_idx] <= amdf[orig_dip] * 1.1:
            best_lag = cand_lag
    dip = best_lag - lag_min
    lag = lag_min + dip
    f0 = sr / lag if lag > 0 else 0.0
    ref = float(np.max(amdf)) if np.max(amdf) > 1e-9 else 1.0
    conf = float(1.0 - amdf[dip] / ref)
    if dip == 0 or dip == len(amdf) - 1:
        conf *= 0.3
    return f0, conf


def _raw_frames(clip: AudioClip, frame_ms: float, overlap_pct: float) -> FrameSet:
    # Pitch detectors need *unwindowed* frames: autocorrelation/AMDF are
    # sensitive to the near-zero taper a Hamming/Hanning window puts at the
    # frame edges, which otherwise biases them toward spuriously long lags.
    return frame_signal(clip, frame_ms=frame_ms, overlap_pct=overlap_pct, window="boxcar")


def pitch_autocorrelation(
    clip: AudioClip, frame_ms: float = 20.0, overlap_pct: float = 50.0, fmin: float = 65.0, fmax: float = 400.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame F0 via the autocorrelation method: pick the lag (in the
    plausible pitch-period range) with the highest autocorrelation peak."""
    fs = _raw_frames(clip, frame_ms, overlap_pct)
    lag_min, lag_max = max(int(fs.sr / fmax), 1), int(fs.sr / fmin)
    out = [_autocorr_pitch_frame(f, fs.sr, lag_min, lag_max) for f in fs.frames]
    f0 = np.array([o[0] for o in out], dtype=np.float32)
    conf = np.array([o[1] for o in out], dtype=np.float32)
    times = np.arange(len(f0)) * fs.hop_length / fs.sr
    return times, f0, conf


def pitch_amdf(
    clip: AudioClip, frame_ms: float = 20.0, overlap_pct: float = 50.0, fmin: float = 65.0, fmax: float = 400.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame F0 via AMDF (Average Magnitude Difference Function): pick the
    lag with the deepest dip (most similar shifted copy of the frame)."""
    fs = _raw_frames(clip, frame_ms, overlap_pct)
    lag_min, lag_max = max(int(fs.sr / fmax), 1), int(fs.sr / fmin)
    out = [_amdf_pitch_frame(f, fs.sr, lag_min, lag_max) for f in fs.frames]
    f0 = np.array([o[0] for o in out], dtype=np.float32)
    conf = np.array([o[1] for o in out], dtype=np.float32)
    times = np.arange(len(f0)) * fs.hop_length / fs.sr
    return times, f0, conf


def fix_octave_errors(f0: np.ndarray, conf: np.ndarray, conf_threshold: float = 0.3, ratio_tol: float = 0.12) -> np.ndarray:
    """Mitigate isolated pitch doubling/halving: for each reliable frame, if
    it disagrees by roughly a factor of 2 (or 1/2) with the GLOBAL median of
    every reliable frame in the whole clip, snap it back onto that octave.

    This median is a single whole-clip scalar, not a local/neighboring-frame
    statistic -- it can misfire on a genuine sustained pitch excursion (e.g.
    question intonation lasting many consecutive frames) that happens to sit
    near 2x the clip's overall median, since nothing here distinguishes an
    isolated single-frame artifact from a real, sustained contour change. It
    is best suited to cleaning up occasional one-off octave slips, not to
    tracking genuine pitch movement."""
    fixed = f0.copy()
    reliable = conf >= conf_threshold
    if not np.any(reliable):
        return fixed
    med = float(np.median(fixed[reliable]))
    if med <= 0:
        return fixed
    for i in range(len(fixed)):
        if not reliable[i] or fixed[i] <= 0:
            continue
        ratio = fixed[i] / med
        if abs(ratio - 2.0) < ratio_tol * 2:
            fixed[i] /= 2.0
        elif abs(ratio - 0.5) < ratio_tol * 0.5:
            fixed[i] *= 2.0
    return fixed


def vad_frames(
    clip: AudioClip,
    frame_ms: float = 20.0,
    overlap_pct: float = 50.0,
    energy_margin_db: float = 12.0,
    zcr_threshold: float = 0.20,
    pitch_conf_threshold: float = 0.25,
) -> dict:
    """Rule-based voice activity detection using relative energy, pitch
    reliability (autocorrelation confidence), and zero-crossing rate. A frame
    counts as active speech if its energy clears a margin above the noise
    floor AND it looks either voiced (a reliable pitch is detected) or like
    an unvoiced consonant (energy above threshold but a noisy, high
    zero-crossing rate rather than a periodic waveform)."""
    fs = _raw_frames(clip, frame_ms, overlap_pct)
    hop_length, frame_length = fs.hop_length, fs.frame_length
    # center=False so these frames align exactly with the (also non-centered)
    # frames pitch_autocorrelation uses. librosa's default center=True pads
    # and centers each frame, which offsets it by half a frame from the raw
    # frames used for pitch -- left uncorrected, every feature at index i
    # would describe a different slice of audio, worst at exactly the
    # speech/pause and voiced/unvoiced transitions VAD cares most about.
    rms = librosa.feature.rms(y=clip.y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    zcr = librosa.feature.zero_crossing_rate(
        clip.y, frame_length=frame_length, hop_length=hop_length, center=False
    )[0]
    times = np.arange(len(rms)) * hop_length / clip.sr

    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))
    noise_floor_db = float(np.percentile(rms_db, 10))
    threshold_db = noise_floor_db + energy_margin_db

    _, _, conf_ac = pitch_autocorrelation(clip, frame_ms=frame_ms, overlap_pct=overlap_pct)
    n = min(len(times), len(zcr), len(conf_ac))
    times, rms_db, zcr, conf_ac = times[:n], rms_db[:n], zcr[:n], conf_ac[:n]

    relative_energy_db = rms_db - noise_floor_db
    voiced_like = conf_ac >= pitch_conf_threshold
    # Unvoiced consonants (s/f/sh, stops) are broadband and noise-like, which
    # means a HIGH zero-crossing rate -- not low. Low ZCR marks periodic
    # (voiced) content, already covered by voiced_like above.
    unvoiced_like = zcr > zcr_threshold
    active = (rms_db > threshold_db) & (voiced_like | unvoiced_like)
    if len(active) >= 3:
        active = sp_signal.medfilt(active.astype(int), kernel_size=3).astype(bool)

    return {
        "times": times,
        "active": active,
        "relative_energy_db": relative_energy_db,
        "zcr": zcr,
        "pitch_confidence": conf_ac,
        "threshold_db": threshold_db,
        "noise_floor_db": noise_floor_db,
        "frame_length": frame_length,
        "hop_length": hop_length,
    }


# ---------------------------------------------------------------------------
# LPC analysis via the Levinson-Durbin recursion
#
#   E(0) = phi[0]
#   k_i  = { phi[i] - sum_{j=1}^{i-1} a_j^(i-1) phi[i-j] } / E(i-1),  1<=i<=P
#   a_i^(i) = k_i
#   a_j^(i) = a_j^(i-1) - k_i * a_{i-j}^(i-1),  1<=j<=i-1
#   E(i) = (1 - k_i^2) E(i-1)
#
# matching the analysis/synthesis pair x(n) -> [A(z)] -> e(n) -> [1/A(z)] -> s^(n),
# with the vocal-tract model V(z) = B / A(z), A(z) = 1 - sum_{k=1}^{P} a_k z^-k.
# ---------------------------------------------------------------------------


def autocorrelate_frame(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation phi[0..max_lag] of a single (already-windowed) frame."""
    n = len(frame)
    max_lag = min(max_lag, n - 1)
    return np.array([np.dot(frame[: n - lag], frame[lag:]) for lag in range(max_lag + 1)])


def levinson_durbin(phi: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for LPC coefficients from an autocorrelation sequence phi[0..order]
    via the Levinson-Durbin recursion (see the module-level derivation above).

    Returns (a, k, E):
      a -- LPC coefficients a[1..order] s.t. x[n] ~= sum_j a[j-1] * x[n-j]
      k -- reflection (PARCOR) coefficients, one produced per recursion step
      E -- prediction error energy E[0..order]; E[0] = phi[0], non-increasing
    """
    order = min(order, len(phi) - 1)
    E = np.zeros(order + 1)
    a = np.zeros(order + 1)  # a[0] unused (an implicit leading 1); a[1..order] returned
    k = np.zeros(order + 1)  # k[0] unused
    E[0] = phi[0]
    if phi[0] <= 1e-12:
        return a[1:], k[1:], E
    for i in range(1, order + 1):
        acc = phi[i] - float(np.dot(a[1:i], phi[i - 1:0:-1]))
        k_i = acc / E[i - 1] if E[i - 1] > 1e-12 else 0.0
        k_i = max(-0.999999, min(0.999999, k_i))  # keep the all-pole filter stable
        a_prev = a.copy()
        a[i] = k_i
        for j in range(1, i):
            a[j] = a_prev[j] - k_i * a_prev[i - j]
        k[i] = k_i
        E[i] = (1 - k_i ** 2) * E[i - 1]
    return a[1:], k[1:], E


@dataclass
class LPCResult:
    order: int
    a: np.ndarray        # LPC coefficients a[1..P]
    k: np.ndarray         # reflection coefficients
    gain: float           # G = sqrt(final prediction error energy)
    frame: np.ndarray     # the windowed analysis frame, x(n)
    residual: np.ndarray  # e(n) = x(n) - sum a_k x(n-k), i.e. x(n) through A(z)
    sr: int


def lpc_analyze_frame(frame: np.ndarray, sr: int, order: int, window: str = "hamming") -> LPCResult:
    """Classic autocorrelation-method LPC analysis of one frame: window it,
    estimate its autocorrelation, run Levinson-Durbin, and compute the
    residual e(n) by passing the frame through the analysis filter A(z)."""
    n = len(frame)
    win = np.ones(n) if window == "boxcar" else sp_signal.get_window(window, n, fftbins=True)
    windowed = (frame * win).astype(np.float64)
    phi = autocorrelate_frame(windowed, order)
    a, k, E = levinson_durbin(phi, order)
    gain = float(np.sqrt(max(E[-1], 0.0)))
    b = np.concatenate(([1.0], -a))  # e(n) = x(n) - sum a_k x(n-k) is a pure FIR (MA) filter
    residual = sp_signal.lfilter(b, [1.0], windowed)
    return LPCResult(order=len(a), a=a, k=k, gain=gain, frame=windowed, residual=residual, sr=sr)


def lpc_spectrum(result: LPCResult, n_freqs: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """The vocal-tract (AR) spectrum V(w) = G / A(e^jw), as 20*log10|V(w)| in dB,
    matching the "LPC SPECTRUM" / "AR-spectrum" curves derived from V(z)=B/A(z)."""
    w = np.linspace(0, np.pi, n_freqs)
    A = np.ones(n_freqs, dtype=complex)
    for idx, a_k in enumerate(result.a, start=1):
        A -= a_k * np.exp(-1j * w * idx)
    V = result.gain / A
    freqs = w * result.sr / (2 * np.pi)
    mag_db = 20 * np.log10(np.maximum(np.abs(V), 1e-12))
    return freqs, mag_db


def lpc_ola_signal(
    clip: AudioClip,
    order: int,
    frame_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    window: str = "hamming",
) -> dict:
    """Run per-frame Levinson-Durbin LPC analysis across an entire clip -- i.e.
    put the Levinson-Durbin recursion "inside" the OLA program -- and
    reassemble the per-frame residuals e(n) and the per-frame
    synthesis-filter reconstructions into two continuous signals via weighted
    OLA. Reconstruction here applies 1/A(z) to each frame's OWN residual, so
    it recovers the (windowed) input essentially exactly regardless of order
    -- that only validates the analysis/synthesis pipeline is self-consistent,
    it is not evidence that any one order reconstructs "better"; the
    order-dependent quantity worth comparing is the residual energy (how much
    the all-pole model whitens the frame), which is monotonically
    non-increasing in order by construction of the recursion."""
    overlap_pct = max(0.0, (1 - frame_shift_ms / frame_ms) * 100)
    raw = frame_signal(clip, frame_ms=frame_ms, overlap_pct=overlap_pct, window="boxcar")
    per_frame = [lpc_analyze_frame(f, clip.sr, order, window=window) for f in raw.frames]
    win = sp_signal.get_window(window, raw.frame_length, fftbins=True) if raw.frame_length else np.array([])

    if per_frame:
        residual_stack = np.stack([r.residual for r in per_frame])
        recon_frames = []
        for r in per_frame:
            a_full = np.concatenate(([1.0], -r.a))
            recon_frames.append(sp_signal.lfilter([1.0], a_full, r.residual))
        recon_stack = np.stack(recon_frames)
    else:
        residual_stack = np.zeros((0, raw.frame_length))
        recon_stack = np.zeros((0, raw.frame_length))

    residual_fs = FrameSet(frames=residual_stack, window=win, frame_length=raw.frame_length, hop_length=raw.hop_length, sr=clip.sr)
    recon_fs = FrameSet(frames=recon_stack, window=win, frame_length=raw.frame_length, hop_length=raw.hop_length, sr=clip.sr)
    residual_signal, _ = ola_reconstruct(residual_fs, weighted=True)
    recon_signal, _ = ola_reconstruct(recon_fs, weighted=True)

    gains = np.array([r.gain for r in per_frame])
    return {
        "per_frame": per_frame,
        "residual_signal": residual_signal,
        "recon_signal": recon_signal,
        "frame_length": raw.frame_length,
        "hop_length": raw.hop_length,
        "overlap_pct": overlap_pct,
        "mean_residual_energy": float(np.mean(gains ** 2)) if len(gains) else 0.0,
    }


def extract_frame(clip: AudioClip, t_center: float, frame_length: int) -> np.ndarray:
    """A single raw (unwindowed) frame_length-sample frame centered on t_center seconds."""
    start = max(0, min(int(round(t_center * clip.sr)) - frame_length // 2, len(clip.y) - frame_length))
    start = max(0, start)
    f = clip.y[start: start + frame_length]
    if len(f) < frame_length:
        f = np.pad(f, (0, frame_length - len(f)))
    return f.astype(np.float64)


def pick_representative_frames(clip: AudioClip, frame_length: int) -> dict:
    """The longest voiced/unvoiced/background segment's center frame, via the
    Homework #2 classify_segments() classifier -- a reproducible way to pick
    frames "of different characteristics" (e.g. for an LPC/AR-spectrum demo)
    without picking them by eye."""
    _, _, segments, _ = classify_segments(clip)
    picks: dict = {}
    for code, key in [(2, "voiced"), (1, "unvoiced"), (0, "background")]:
        segs = [s for s in segments if s.code == code]
        if not segs:
            continue
        longest = max(segs, key=lambda s: s.end - s.start)
        mid = (longest.start + longest.end) / 2
        picks[key] = {
            "t_center": mid,
            "duration": longest.end - longest.start,
            "frame": extract_frame(clip, mid, frame_length),
        }
    return picks


# ---------------------------------------------------------------------------
# Chapter 6: phase unwrapping (homomorphic / complex-cepstrum analysis)
#
#   Modulo-2pi unwrapper (discrete, "Recall" slide):
#     r[0] = 0; for k > 0:
#       S1: if  PV[X(k)] - PV[X(k-1)] >  2pi - eps  then r[k] = r[k-1] - 1
#       S2: if  PV[X(k)] - PV[X(k-1)] < -(2pi - eps) then r[k] = r[k-1] + 1
#       S3: otherwise                                    r[k] = r[k-1]
#     angle X(k) = PV[X(k)] + 2*pi*r[k]
#
#   Phase-derivative-based unwrapper ("Phase Derivative-based Phase
#   Unwrapper" slides): the continuous phase derivative d(angle X)/d(omega)
#   has no 2pi jumps (only the principal value does), so integrating it
#   (trapezoidal rule) from the already-unwrapped theta(w_{k-1}) predicts
#   theta(w_k); the integer q(w_k) is then chosen to minimize
#     E(k) = | (PV[X(w_k)] + 2*pi*q(w_k)) - theta_hat(w_k) |.
# ---------------------------------------------------------------------------


def phase_unwrap_modulo(phase: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Discrete modulo-2pi phase unwrapper (Chapter6 p.8): given the
    principal-value phase PV[X(k)] in (-pi, pi], find integer r[k] (r[0]=0)
    so that PV[X(k)] + 2*pi*r[k] is continuous, by comparing each adjacent
    jump against +-(2*pi - eps).

    The slide's own note ("threshold eps? -> should be small enough") is a
    real design trade-off, not a throwaway remark: a genuine wrap's raw
    jump is 2*pi minus whatever the true phase actually moved between two
    adjacent frequency samples (never exactly 2*pi), so eps must be LARGER
    than that per-step drift or real wraps get silently missed -- e.g. for
    a pure delay of D samples sampled at n_freqs points over [0, pi], the
    per-step drift is about D*pi/n_freqs, and eps=1e-6 (an eps that is
    itself extremely "small") fails to catch any wrap at all, while
    eps=0.1 matches both np.unwrap and the analytic truth exactly. At the
    same time eps must stay well below 2*pi so a large but genuine
    (non-wrap) jump isn't mistaken for noise -- "small enough" means small
    relative to 2*pi, not small in absolute terms."""
    phase = np.asarray(phase, dtype=np.float64)
    r = np.zeros(len(phase))
    threshold = 2 * np.pi - eps
    for k in range(1, len(phase)):
        diff = phase[k] - phase[k - 1]
        if diff > threshold:
            r[k] = r[k - 1] - 1
        elif diff < -threshold:
            r[k] = r[k - 1] + 1
        else:
            r[k] = r[k - 1]
    return phase + 2 * np.pi * r


def _phase_and_derivative(x: np.ndarray, n_freqs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PV[X(w)] (principal value) and the continuous phase derivative
    d(angle X)/d(omega), both over w in [0, pi]. The derivative is computed
    analytically rather than by differencing the (wrapped) phase: if
    x[n] <-> X(e^jw), then -j*n*x[n] <-> dX/dw, and
    d(angle X)/dw = Im[ (dX/dw) / X(w) ] -- well-defined (no 2pi jump)
    wherever X(w) != 0."""
    n = len(x)
    w = np.linspace(0, np.pi, n_freqs)
    nn = np.arange(n)
    # DTFT at arbitrary (non-FFT-grid) frequencies via direct summation --
    # fine at these report sizes (n_freqs x n <= a few hundred thousand).
    phase_mat = np.exp(-1j * np.outer(w, nn))
    X = phase_mat @ x
    dXdw = phase_mat @ (-1j * nn * x)
    safe_X = np.where(np.abs(X) < 1e-12, 1e-12, X)
    deriv = np.imag(dXdw / safe_X)
    pv = np.angle(X)
    return w, pv, deriv


def phase_unwrap_derivative(x: np.ndarray, n_freqs: int = 512) -> dict:
    """Phase-derivative-based phase unwrapper (Chapter6 p.9-10): combines the
    principal value PV[X(w)] with the continuous phase derivative to predict
    each next unwrapped value theta_hat(w_k) by trapezoidal integration of
    the derivative from w_{k-1} to w_k, then picks the integer q(w_k) that
    minimizes |PV[X(w_k)] + 2*pi*q(w_k) - theta_hat(w_k)| (equivalently,
    rounds (theta_hat - PV) / (2*pi) to the nearest integer)."""
    w, pv, deriv = _phase_and_derivative(x, n_freqs)
    theta = np.zeros(n_freqs)
    theta[0] = pv[0]
    for k in range(1, n_freqs):
        theta_hat = theta[k - 1] + (w[k] - w[k - 1]) / 2 * (deriv[k] + deriv[k - 1])
        q = round((theta_hat - pv[k]) / (2 * np.pi))
        theta[k] = pv[k] + 2 * np.pi * q
    return {"freqs": w, "principal_value": pv, "derivative": deriv, "unwrapped": theta}
