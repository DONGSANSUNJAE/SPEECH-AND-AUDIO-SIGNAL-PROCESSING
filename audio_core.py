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
    ("한국어 · 선희 (여)", "ko-KR-SunHiNeural"),
    ("한국어 · 인준 (남)", "ko-KR-InJoonNeural"),
    ("English · Jenny", "en-US-JennyNeural"),
    ("English · Guy", "en-US-GuyNeural"),
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
    return merged


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
