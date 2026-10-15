"""Audio Lab — speech download, playback, and analysis."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from audio_core import (
    DEFAULT_TTS_TEXT,
    DEMOS,
    SAMPLES,
    VOICES,
    AudioClip,
    analyze,
    classify_segments,
    download_sample,
    download_url,
    fix_octave_errors,
    frame_signal,
    list_downloads,
    load_audio_bytes,
    load_audio_file,
    lpc_analyze_frame,
    lpc_spectrum,
    make_demo,
    ola_reconstruct,
    pitch_amdf,
    pitch_autocorrelation,
    resample_clip,
    save_clip_wav,
    synthesize_speech,
    to_wav_bytes,
    transcribe,
    vad_frames,
    window_overlap_profile,
)

INK = "#1C1916"
MUTED = "#6A635B"
NAVY = "#1B3A6B"
PAPER = "#FFFCFA"
LINE = "#E2D9CC"
FONT = "Pretendard, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
CHART_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "audio_lab"},
}


def configure_page() -> None:
    st.set_page_config(page_title="음성 분석", page_icon=" ", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
        <style>
        html, body, .stApp, p, h1, h2, h3, h4, label, button, input, textarea {
            font-family: Pretendard, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif !important;
        }
        .material-icons, .material-symbols-outlined, .material-symbols-rounded,
        span[translate="no"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            letter-spacing: normal !important;
            font-feature-settings: "liga" 1 !important;
        }
        .stApp { background: #F4EFE7; color: #1C1916; }
        [data-testid="stToolbar"] { background: transparent; }
        .stDeployButton, [data-testid="stDeployButton"], footer, #MainMenu { display: none !important; }
        [data-testid="stDecoration"] { display: none; }
        header[data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"] { display: none !important; }
        section[data-testid="stSidebar"] {
            background: #FFFCFA; border-right: 1px solid #E2D9CC;
            min-width: 340px !important; max-width: 360px !important;
        }
        section[data-testid="stSidebar"] > div { background: #FFFCFA; }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.1rem 1.25rem 1.6rem; }
        .block-container { padding: 2.2rem 2rem 2.6rem; max-width: 1120px; }
        .section {
            background: #FFFCFA; border: 1px solid #E2D9CC; border-radius: 14px;
            padding: 18px 20px 12px; margin: 0 0 16px;
        }
        .section h3, .sidebar-title {
            font-size: 0.8rem; font-weight: 650; margin: 0 0 12px; color: #6A635B;
        }
        .stats { width: 100%; border-collapse: collapse; margin: 0 0 8px; }
        .stats th, .stats td { text-align: left; padding: 8px 4px; border-bottom: 1px solid #EEE6DA; }
        .stats th { color: #6A635B; font-size: 0.78rem; font-weight: 500; width: 26%; }
        .stats td { font-size: 0.92rem; font-weight: 600; word-break: break-all; color: #1C1916; }
        .empty { color: #6A635B; padding: 88px 12px; text-align: center; line-height: 1.65; }
        .empty strong { display: block; color: #1C1916; font-size: 1.05rem; font-weight: 650; margin-bottom: 8px; }
        .empty span { font-size: 0.92rem; }
        [data-testid="stWidgetLabel"] p, label {
            font-size: 0.82rem !important; font-weight: 500 !important; color: #6A635B !important;
        }
        .stButton > button {
            width: 100%; border-radius: 10px; font-weight: 600;
            white-space: nowrap; min-height: 2.5rem;
            padding: 0.5rem 0.9rem; border: 1px solid #1B3A6B;
            background: #1B3A6B; color: #fff;
        }
        .stButton > button:hover { background: #163057; border-color: #163057; color: #fff; }
        textarea, [data-baseweb="input"] input, [data-baseweb="base-input"] {
            border-radius: 10px !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background: transparent; border: 1px solid #E2D9CC; color: #6A635B;
            border-radius: 999px; padding: 7px 14px; font-weight: 600; font-size: 0.86rem;
        }
        .stTabs [aria-selected="true"] { background: #1B3A6B; color: #fff !important; border-color: #1B3A6B; }
        div[data-testid="stAudio"] { margin: 10px 0 2px; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #8A8278 !important; font-size: 0.78rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chart(title: str, height: int, t0: float, t1: float) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK, family=FONT, weight=600)),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=MUTED, family=FONT, size=12),
        margin=dict(l=56, r=16, t=48, b=48),
        height=height,
        xaxis=dict(
            title=dict(text="시간 (초)", font=dict(size=12, color=MUTED)),
            range=[t0, t1],
            gridcolor=LINE,
            zeroline=False,
            tickfont=dict(size=11, color=MUTED),
            rangeslider=dict(visible=True, thickness=0.07, bgcolor="#F3EDE3"),
        ),
        yaxis=dict(gridcolor=LINE, zeroline=False, tickfont=dict(size=11, color=MUTED)),
    )
    return fig


def plotly_waveform(clip: AudioClip, t0: float, t1: float) -> go.Figure:
    fig = _chart("파형", 280, t0, t1)
    fig.add_scatter(
        x=np.arange(len(clip.y)) / clip.sr,
        y=clip.y,
        mode="lines",
        line=dict(color=NAVY, width=1.2),
        hovertemplate="%{x:.3f}초 · 진폭 %{y:.3f}<extra></extra>",
    )
    fig.update_yaxes(title="진폭")
    return fig


def plotly_mel(clip: AudioClip, n_fft: int, hop: int, n_mels: int, fmin: int, fmax: int, t0: float, t1: float, y0: float, y1: float) -> go.Figure:
    fmax = min(fmax, clip.sr // 2)
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=clip.y, sr=clip.sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0),
        ref=np.max,
    )
    fig = _chart("멜 스펙트로그램", 480, t0, t1)
    fig.add_heatmap(
        z=S,
        x=librosa.times_like(S, sr=clip.sr, hop_length=hop),
        y=librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax),
        colorscale="Hot",
        colorbar=dict(title="dB", thickness=14),
        hovertemplate="%{x:.3f}초 · %{y:.0f} Hz · %{z:.1f} dB<extra></extra>",
    )
    fig.update_yaxes(title="멜 주파수 (Hz)", range=[y0, y1])
    return fig


def plotly_stft(clip: AudioClip, n_fft: int, hop: int, t0: float, t1: float, y0: float, y1: float) -> go.Figure:
    S = librosa.amplitude_to_db(np.abs(librosa.stft(clip.y, n_fft=n_fft, hop_length=hop)), ref=np.max)
    fig = _chart("선형 스펙트로그램 (STFT)", 480, t0, t1)
    fig.add_heatmap(
        z=S,
        x=librosa.times_like(S, sr=clip.sr, hop_length=hop),
        y=librosa.fft_frequencies(sr=clip.sr, n_fft=n_fft),
        colorscale="Hot",
        colorbar=dict(title="dB", thickness=14),
        hovertemplate="%{x:.3f}초 · %{y:.0f} Hz · %{z:.1f} dB<extra></extra>",
    )
    fig.update_yaxes(title="주파수 (Hz)", range=[y0, y1])
    return fig


def plotly_spectrum(clip: AudioClip) -> go.Figure:
    n = len(clip.y)
    mag = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(clip.y * np.hanning(n))), 1e-12))
    mag -= np.max(mag)
    fig = _chart("평균 스펙트럼", 340, 0, 1)
    fig.update_xaxes(title="주파수 (Hz)", range=[0, clip.sr / 2], rangeslider=None)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.add_scatter(
        x=np.fft.rfftfreq(n, d=1 / clip.sr),
        y=mag,
        mode="lines",
        line=dict(color="#A8481C", width=1.4),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>",
    )
    fig.update_yaxes(title="크기 (dB)", range=[-80, 3])
    return fig


def plotly_pitch(clip: AudioClip, hop: int) -> go.Figure:
    f0 = librosa.yin(clip.y, fmin=65, fmax=400, sr=clip.sr, hop_length=hop)
    times = librosa.times_like(f0, sr=clip.sr, hop_length=hop)
    fig = _chart("음높이 (F0)", 340, 0, clip.duration)
    fig.add_scatter(
        x=times,
        y=np.where(np.isfinite(f0), f0, np.nan),
        mode="lines+markers",
        line=dict(color=NAVY, width=2),
        marker=dict(size=4),
        hovertemplate="%{x:.3f}초 · %{y:.0f} Hz<extra></extra>",
    )
    fig.update_yaxes(title="F0 (Hz)", range=[60, 420])
    return fig


SEGMENT_COLORS = {0: "#C9BFAE", 1: "#C97A3B", 2: "#1B3A6B"}


def plotly_segments(clip: AudioClip, segments: list) -> go.Figure:
    fig = _chart("파형 + 자동 분류 구간", 300, 0.0, max(clip.duration, 0.05))
    for seg in segments:
        fig.add_vrect(
            x0=seg.start, x1=seg.end,
            fillcolor=SEGMENT_COLORS.get(seg.code, "#CCCCCC"), opacity=0.28, line_width=0,
        )
    fig.add_scatter(
        x=np.arange(len(clip.y)) / clip.sr,
        y=clip.y,
        mode="lines",
        line=dict(color=INK, width=1.0),
        hovertemplate="%{x:.3f}초 · 진폭 %{y:.3f}<extra></extra>",
    )
    fig.update_yaxes(title="진폭")
    fig.update_layout(showlegend=False)
    return fig


def plotly_energy_zcr(times: np.ndarray, diag: dict) -> go.Figure:
    t0 = float(times[0]) if len(times) else 0.0
    t1 = float(times[-1]) if len(times) else 0.05
    fig = _chart("프레임별 에너지(RMS) · ZCR", 280, t0, t1)
    fig.add_scatter(
        x=times, y=diag["rms_db"], mode="lines", name="에너지 (dB)",
        line=dict(color=NAVY, width=1.4),
        hovertemplate="%{x:.3f}초 · 에너지 %{y:.1f} dB<extra></extra>",
    )
    fig.add_hline(
        y=diag["threshold_db"],
        line=dict(color="#A8481C", width=1.2, dash="dash"),
        annotation_text=f"발화 판정 임계값 {diag['threshold_db']:.1f} dB",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#A8481C"),
    )
    fig.add_scatter(
        x=times, y=diag["zcr"], mode="lines", name="ZCR", yaxis="y2",
        line=dict(color="#C97A3B", width=1.0),
        hovertemplate="%{x:.3f}초 · ZCR %{y:.3f}<extra></extra>",
    )
    zcr_max = float(np.max(diag["zcr"])) if len(diag["zcr"]) else 0.5
    fig.update_layout(
        yaxis=dict(title="에너지 (dB)", gridcolor=LINE, zeroline=False),
        yaxis2=dict(title="ZCR", overlaying="y", side="right", range=[0, max(0.5, zcr_max * 1.1)], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plotly_spectrum_compare(entries: list[tuple[str, AudioClip, str]]) -> go.Figure:
    fig = _chart("스펙트럼 비교", 340, 0, 1)
    fig.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    max_sr = max(clip.sr for _, clip, _ in entries)
    for label, clip, color in entries:
        n = len(clip.y)
        mag = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(clip.y * np.hanning(n))), 1e-12))
        mag -= np.max(mag)
        freqs = np.fft.rfftfreq(n, d=1 / clip.sr)
        fig.add_scatter(
            x=freqs, y=mag, mode="lines", name=label,
            line=dict(color=color, width=1.4),
            hovertemplate=f"{label} · " + "%{x:.0f} Hz · %{y:.1f} dB<extra></extra>",
        )
    fig.update_xaxes(title="주파수 (Hz)", range=[0, max_sr / 2])
    fig.update_yaxes(title="크기 (dB)", range=[-80, 3])
    return fig


def plotly_spectrogram_fixed(clip: AudioClip, freq_range: float, title: str, height: int = 220) -> go.Figure:
    """Linear STFT spectrogram with a fixed frequency range/color scale so several
    sample rates can be compared side by side on identical axes."""
    n_fft = 1024
    hop = 128
    S = librosa.amplitude_to_db(np.abs(librosa.stft(clip.y, n_fft=n_fft, hop_length=hop)), ref=np.max)
    times = librosa.times_like(S, sr=clip.sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=clip.sr, n_fft=n_fft)
    fig = _chart(title, height, 0.0, max(clip.duration, 0.05))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.add_heatmap(
        z=S, x=times, y=freqs, zmin=-80, zmax=0, colorscale="Hot",
        colorbar=dict(title="dB", thickness=10),
        hovertemplate="%{x:.3f}초 · %{y:.0f} Hz · %{z:.1f} dB<extra></extra>",
    )
    fig.update_yaxes(title="주파수 (Hz)", range=[0, freq_range])
    return fig


def plotly_window_sum(profile: np.ndarray, sr: int, hop_length: int, window_label: str) -> go.Figure:
    times = np.arange(len(profile)) / sr
    fig = _chart(f"윈도우 겹침 합 (COLA 확인) — {window_label}", 220, 0.0, float(times[-1]) if len(times) else 0.05)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.add_scatter(
        x=times, y=profile, mode="lines", line=dict(color=NAVY, width=1.2),
        hovertemplate="%{x:.3f}초 · 합 %{y:.3f}<extra></extra>",
    )
    fig.update_yaxes(title="윈도우 합")
    return fig


def plotly_ola_compare(original: np.ndarray, reconstructed: np.ndarray, sr: int, title: str) -> go.Figure:
    n = min(len(original), len(reconstructed))
    t = np.arange(n) / sr
    fig = _chart(title, 260, 0.0, float(t[-1]) if n else 0.05)
    fig.add_scatter(x=t, y=original[:n], mode="lines", name="입력", line=dict(color=MUTED, width=1.0))
    fig.add_scatter(x=t, y=reconstructed[:n], mode="lines", name="출력(재합성)", line=dict(color=NAVY, width=1.0))
    fig.update_yaxes(title="진폭")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def plotly_ola_error(original: np.ndarray, reconstructed: np.ndarray, sr: int) -> go.Figure:
    n = min(len(original), len(reconstructed))
    t = np.arange(n) / sr
    err = original[:n] - reconstructed[:n]
    fig = _chart("오차 (입력 − 출력)", 180, 0.0, float(t[-1]) if n else 0.05)
    fig.add_scatter(x=t, y=err, mode="lines", line=dict(color="#A8481C", width=1.0))
    fig.update_yaxes(title="오차")
    return fig


def plotly_pitch_compare(t_ac, f0_ac, conf_ac, t_amdf, f0_amdf, conf_amdf, conf_th: float) -> go.Figure:
    f0_ac_masked = np.where(conf_ac >= conf_th, f0_ac, np.nan)
    f0_amdf_masked = np.where(conf_amdf >= conf_th, f0_amdf, np.nan)
    t_end = max(float(t_ac[-1]) if len(t_ac) else 0.05, float(t_amdf[-1]) if len(t_amdf) else 0.05)
    fig = _chart("피치(F0) 비교: 자기상관 vs AMDF", 320, 0.0, t_end)
    fig.add_scatter(x=t_ac, y=f0_ac_masked, mode="lines+markers", name="자기상관",
                     line=dict(color=NAVY, width=1.6), marker=dict(size=3))
    fig.add_scatter(x=t_amdf, y=f0_amdf_masked, mode="lines+markers", name="AMDF",
                     line=dict(color="#C97A3B", width=1.6), marker=dict(size=3))
    fig.update_yaxes(title="F0 (Hz)", range=[0, 420])
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def plotly_vad(clip: AudioClip, vad: dict) -> go.Figure:
    fig = _chart("VAD 결과 (파형 + 음성 활성 구간)", 300, 0.0, max(clip.duration, 0.05))
    times, active = vad["times"], vad["active"]
    hop_s = vad["hop_length"] / clip.sr
    in_run = False
    run_start = 0.0
    for i, a in enumerate(active):
        if a and not in_run:
            in_run, run_start = True, times[i] - hop_s / 2
        elif not a and in_run:
            in_run = False
            fig.add_vrect(x0=max(0.0, run_start), x1=times[i] - hop_s / 2, fillcolor=NAVY, opacity=0.22, line_width=0)
    if in_run:
        fig.add_vrect(x0=max(0.0, run_start), x1=times[-1] + hop_s / 2, fillcolor=NAVY, opacity=0.22, line_width=0)
    fig.add_scatter(
        x=np.arange(len(clip.y)) / clip.sr, y=clip.y, mode="lines",
        line=dict(color=INK, width=1.0), hovertemplate="%{x:.3f}초 · 진폭 %{y:.3f}<extra></extra>",
    )
    fig.update_yaxes(title="진폭")
    fig.update_layout(showlegend=False)
    return fig


def plotly_vad_features(vad: dict) -> go.Figure:
    times = vad["times"]
    fig = _chart("VAD 특징: 상대 에너지 · ZCR · 피치 신뢰도", 260, 0.0, float(times[-1]) if len(times) else 0.05)
    fig.add_scatter(x=times, y=vad["relative_energy_db"], mode="lines", name="상대 에너지(dB)",
                     line=dict(color=NAVY, width=1.3))
    fig.add_scatter(x=times, y=vad["pitch_confidence"] * 40, mode="lines", name="피치 신뢰도(×40, 참고용)",
                     line=dict(color="#7A9A6B", width=1.0))
    fig.add_scatter(x=times, y=vad["zcr"] * 100, mode="lines", name="ZCR(×100)", yaxis="y2",
                     line=dict(color="#C97A3B", width=1.0))
    fig.add_hline(y=vad["threshold_db"] - vad["noise_floor_db"], line=dict(color="#A8481C", width=1.0, dash="dash"))
    fig.update_layout(
        yaxis=dict(title="dB"),
        yaxis2=dict(title="ZCR ×100", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plotly_lpc_signal_error(result) -> go.Figure:
    t = np.arange(len(result.frame)) / result.sr
    fig = _chart(f"LPC 분석 프레임: Signal x(n) vs Error e(n)  (P={result.order})", 260, 0.0, float(t[-1]) if len(t) else 0.05)
    fig.add_scatter(x=t, y=result.frame, mode="lines", name="Signal x(n)", line=dict(color=MUTED, width=1.1))
    fig.add_scatter(x=t, y=result.residual, mode="lines", name="Error e(n)", line=dict(color="#A8481C", width=1.1))
    fig.update_yaxes(title="진폭")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


LPC_ORDER_COLORS = [NAVY, "#C97A3B", "#7A9A6B", "#A8481C", "#8B5FBF"]


def plotly_lpc_spectrum_compare(frame: np.ndarray, sr: int, orders: list[int], highlight_order: int) -> go.Figure:
    fig = _chart("신호 스펙트럼 vs LPC(AR) 스펙트럼 — 차수(P)별 비교", 420, 0, 1)
    fig.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    n = len(frame)
    mag = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(frame * np.hanning(n))), 1e-12))
    mag -= np.max(mag)
    freqs_fft = np.fft.rfftfreq(n, d=1 / sr)
    fig.add_scatter(
        x=freqs_fft, y=mag, mode="lines", name="신호 스펙트럼 (FFT)",
        line=dict(color="#C9BFAE", width=1.4),
        hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>",
    )
    for i, p in enumerate(orders):
        res_p = lpc_analyze_frame(frame, sr, p)
        freqs, mag_db = lpc_spectrum(res_p)
        mag_db = mag_db - np.max(mag_db)
        is_hl = p == highlight_order
        fig.add_scatter(
            x=freqs, y=mag_db, mode="lines", name=f"LPC p={p}" + (" (선택)" if is_hl else ""),
            line=dict(color=LPC_ORDER_COLORS[i % len(LPC_ORDER_COLORS)], width=2.6 if is_hl else 1.3),
            hovertemplate=f"P={p} · " + "%{x:.0f} Hz · %{y:.1f} dB<extra></extra>",
        )
    fig.update_xaxes(title="주파수 (Hz)", range=[0, sr / 2])
    fig.update_yaxes(title="크기 (dB, 각자 최대값 0dB 기준)")
    return fig


def pick_clip():
    mode = st.selectbox(
        "방법",
        ["문장으로 음성 만들기", "마이크 녹음", "내 컴퓨터 파일", "샘플 음성 받기", "주소로 받기", "연습용 신호"],
    )
    clip = None
    if mode == "문장으로 음성 만들기":
        text = st.text_area("읽을 문장", DEFAULT_TTS_TEXT, height=120)
        voice_label = st.selectbox("목소리", [label for label, _ in VOICES])
        if st.button("음성 받기", width="stretch"):
            with st.spinner("음성을 받는 중입니다."):
                clip = load_audio_file(synthesize_speech(text, dict(VOICES)[voice_label]))
    elif mode == "마이크 녹음":
        rec = st.audio_input("짧게 말해 보세요")
        if rec:
            clip = load_audio_bytes(rec.getvalue(), "microphone.wav")
    elif mode == "내 컴퓨터 파일":
        file = st.file_uploader("오디오 파일", type=["wav", "mp3", "flac", "ogg", "m4a", "aac"])
        if file:
            clip = load_audio_bytes(file.read(), file.name)
    elif mode == "샘플 음성 받기":
        label = st.selectbox("공개 낭독 음성", [name for name, _ in SAMPLES])
        if st.button("샘플 받기", width="stretch"):
            with st.spinner("샘플을 받는 중입니다."):
                clip = load_audio_file(download_sample(dict(SAMPLES)[label]))
    elif mode == "주소로 받기":
        url = st.text_input("오디오 주소", placeholder="https://example.com/speech.wav")
        if st.button("받기", width="stretch") and url.strip():
            with st.spinner("파일을 받는 중입니다."):
                clip = load_audio_file(download_url(url))
    else:
        kind = st.selectbox("신호", DEMOS, format_func=lambda x: x[0])
        if st.button("만들기", width="stretch"):
            clip = make_demo(kind[1])

    recent = list_downloads()[:8]
    if recent:
        st.divider()
        chosen = st.selectbox("최근 파일", ["-"] + [p.name for p in recent])
        if chosen != "-" and st.button("다시 열기", width="stretch"):
            clip = load_audio_file(next(p for p in recent if p.name == chosen))
    return clip


def view_params() -> tuple[int, int, int, int, int]:
    with st.expander("스펙트로그램 설정", expanded=False):
        n_mels = st.slider("멜 필터 개수", 16, 256, 128, 8)
        n_fft = st.select_slider("주파수 해상도 (n_fft)", options=[256, 512, 1024, 2048, 4096], value=2048)
        hop = st.select_slider("시간 해상도 (hop)", options=[128, 256, 512, 1024], value=512)
        fmin = int(st.number_input("가장 낮은 주파수 (Hz)", min_value=0, max_value=4000, value=0, step=10))
        fmax = int(st.number_input("가장 높은 주파수 (Hz)", min_value=1000, max_value=24000, value=8000, step=100))
        st.caption("그래프를 드래그하거나 휠로 확대할 수 있습니다. 더블클릭하면 돌아갑니다.")
        return n_fft, hop, n_mels, fmin, fmax
    return 2048, 512, 128, 0, 8000


def render_stats(clip: AudioClip, stats: dict[str, float]) -> None:
    rows = [
        ("파일", clip.name),
        ("길이", f"{clip.duration:.2f}초"),
        ("샘플레이트", f"{clip.sr:,} Hz"),
        ("음량", f"{stats['rms']:.4f}"),
        ("음높이", f"{stats['f0']:.0f} Hz" if stats["f0"] else "—"),
        ("밝기", f"{stats['centroid']:.0f} Hz"),
    ]
    cells = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    st.markdown(f'<table class="stats">{cells}</table>', unsafe_allow_html=True)


def main() -> None:
    configure_page()
    if "clip" not in st.session_state:
        st.session_state.clip = None

    with st.sidebar:
        st.markdown('<p class="sidebar-title">입력</p>', unsafe_allow_html=True)
        new_clip = pick_clip()
        n_fft, hop, n_mels, fmin, fmax = view_params()
    if new_clip is not None:
        st.session_state.clip = new_clip
    clip = st.session_state.clip
    if clip is None:
        st.markdown(
            '<div class="empty"><strong>음성을 넣어 주세요</strong>'
            "<span>왼쪽 입력에서 문장, 녹음, 파일 중 하나를 고르면 됩니다.</span></div>",
            unsafe_allow_html=True,
        )
        return

    stats = analyze(clip, hop)
    st.markdown('<div class="section"><h3>재생</h3>', unsafe_allow_html=True)
    render_stats(clip, stats)
    st.audio(to_wav_bytes(clip), format="audio/wav")
    if st.button("이 음성을 downloads/ 폴더에 저장", width="stretch"):
        saved_path = save_clip_wav(clip)
        st.success(f"저장됨: downloads/{saved_path.name}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>음성 인식 (STT, 로컬 Whisper)</h3>', unsafe_allow_html=True)
    model_size = st.selectbox(
        "모델 크기", ["tiny", "base", "small"], index=1,
        help="크기가 클수록 정확하지만 느립니다. 모델은 처음 고를 때 한 번만 내려받습니다.",
    )
    if st.button("텍스트로 변환", width="stretch"):
        with st.spinner("음성을 텍스트로 변환하는 중입니다 (처음이면 모델 다운로드도 함께 진행돼 더 걸릴 수 있어요)."):
            st.session_state.transcript = transcribe(clip, model_size)
            st.session_state.transcript_key = clip.name
    result = st.session_state.get("transcript")
    if result and st.session_state.get("transcript_key") == clip.name:
        st.caption(f"인식된 언어: {result['language']} (신뢰도 {result['language_probability'] * 100:.0f}%)")
        st.text_area("인식 결과", result["text"], height=100)
        if result["segments"]:
            seg_rows = "".join(
                f"<tr><td>{s:.2f} ~ {e:.2f} 초</td><td>{t}</td></tr>" for s, e, t in result["segments"]
            )
            st.markdown(
                f'<table class="stats"><tr><th>구간</th><th>내용</th></tr>{seg_rows}</table>',
                unsafe_allow_html=True,
            )
        st.download_button(
            "인식 결과 텍스트로 저장",
            data=result["text"],
            file_name=f"{Path(clip.name).stem}_transcript.txt",
            mime="text/plain",
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    fmax_plot = min(fmax, clip.sr // 2)
    duration = max(float(clip.duration), 0.05)
    st.markdown('<div class="section"><h3>파형 · 스펙트로그램</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        t0, t1 = st.slider("보는 시간 (초)", 0.0, duration, (0.0, duration), step=min(0.01, duration / 100))
    with c2:
        y0, y1 = st.slider("보는 주파수 (Hz)", float(fmin), float(fmax_plot), (float(fmin), float(fmax_plot)), step=10.0)
    if t1 - t0 < 0.02:
        t1 = min(duration, t0 + 0.02)
    if y1 - y0 < 20:
        y1 = min(float(fmax_plot), y0 + 20)

    st.plotly_chart(plotly_waveform(clip, t0, t1), use_container_width=True, config=CHART_CONFIG)
    st.plotly_chart(plotly_mel(clip, n_fft, hop, n_mels, fmin, fmax, t0, t1, y0, y1), use_container_width=True, config=CHART_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>스펙트럼 · 음높이</h3>', unsafe_allow_html=True)
    tab_stft, tab_spec, tab_pitch = st.tabs(["선형 스펙트로그램", "평균 스펙트럼", "음높이"])
    with tab_stft:
        st.plotly_chart(plotly_stft(clip, n_fft, hop, t0, t1, y0, y1), use_container_width=True, config=CHART_CONFIG)
    with tab_spec:
        st.plotly_chart(plotly_spectrum(clip), use_container_width=True, config=CHART_CONFIG)
    with tab_pitch:
        st.plotly_chart(plotly_pitch(clip, hop), use_container_width=True, config=CHART_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>샘플레이트 비교 (다운샘플링)</h3>', unsafe_allow_html=True)
    st.caption(f"원본 샘플레이트: {clip.sr:,} Hz. 아래에서 16 kHz / 8 kHz / 2 kHz로 낮췄을 때 소리와 대역폭이 어떻게 달라지는지 들어 보세요.")
    targets = [t for t in (16000, 8000, 2000) if t < clip.sr] or [clip.sr]
    downs = {target: resample_clip(clip, target) for target in targets}
    compare_colors = {16000: NAVY, 8000: "#C97A3B", 2000: "#A8481C"}
    entries = [("원본", clip, MUTED)] + [
        (f"{target:,} Hz", downs[target], compare_colors.get(target, INK)) for target in targets
    ]
    st.plotly_chart(plotly_spectrum_compare(entries), use_container_width=True, config=CHART_CONFIG)
    st.caption("한 그래프에 겹쳐 그린 스펙트럼입니다 — 각 선이 자기 나이퀴스트 주파수에서 뚝 끊기는 걸 한눈에 볼 수 있어요. "
               "아래 카드들의 스펙트로그램도 모두 같은 주파수 축(0~원본 나이퀴스트)과 같은 색 기준으로 그려서, 대역이 잘려나가는 지점을 서로 비교할 수 있습니다.")

    cols = st.columns(len(targets))
    notes = {
        16000: "8 kHz까지의 대역을 보존합니다. 대부분의 자음·마찰음 정보가 남아 원음과 큰 차이를 느끼기 어렵습니다.",
        8000: "4 kHz까지만 남습니다. 전화 음질과 비슷하며, 치찰음(s, sh)의 고역이 다소 뭉개집니다.",
        2000: "1 kHz까지만 남아 대부분의 마찰음·파열음 단서가 사라집니다. 모음은 들리지만 자음 구분이 어려워 알아듣기 힘들어집니다.",
    }
    for col, target in zip(cols, targets):
        with col:
            down = downs[target]
            st.markdown(f"**{target:,} Hz** (나이퀴스트 {target // 2:,} Hz)")
            # Browsers refuse to play very low sample-rate WAVs (e.g. 2 kHz), so
            # upsample only the playback copy; no information is added back.
            playable = resample_clip(down, 8000) if down.sr < 8000 else down
            st.audio(to_wav_bytes(playable), format="audio/wav")
            spec_title = f"스펙트로그램 ({target:,} Hz, 0-{clip.sr // 2:,} Hz 축 고정)"
            st.plotly_chart(
                plotly_spectrogram_fixed(down, freq_range=clip.sr / 2, title=spec_title, height=260),
                use_container_width=True, config=CHART_CONFIG,
            )
            st.caption(notes.get(target, ""))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>발화 구간 자동 분류 (voiced / unvoiced / background)</h3>', unsafe_allow_html=True)
    with st.expander("분류 기준 설정", expanded=False):
        energy_margin = st.slider("에너지 임계값 여유 (dB, 잡음 바닥 대비)", 3.0, 30.0, 12.0, 1.0)
        zcr_th = st.slider("무성음 판정 ZCR 임계값", 0.02, 0.3, 0.12, 0.01)
    times_c, _, segments, diag = classify_segments(clip, energy_margin_db=energy_margin, zcr_threshold=zcr_th)
    st.plotly_chart(plotly_energy_zcr(times_c, diag), use_container_width=True, config=CHART_CONFIG)
    st.caption(
        f"분석 프레임 {diag['frame_ms']:.0f}ms, 이동 간격(hop) {diag['hop_ms']:.0f}ms → 프레임끼리 "
        f"{diag['overlap_pct']:.0f}% 겹쳐서(overlap) 짧은 구간마다 에너지·ZCR을 계산합니다. "
        f"잡음 바닥은 {diag['noise_floor_db']:.1f} dB로 추정, 위 설정의 여유값을 더해 임계값을 정합니다."
    )
    st.plotly_chart(plotly_segments(clip, segments), use_container_width=True, config=CHART_CONFIG)
    rows = "".join(
        f"<tr><td>{seg.start:.2f} ~ {seg.end:.2f} 초</td><td>{seg.label}</td></tr>" for seg in segments
    )
    st.markdown(
        f'<table class="stats"><tr><th>구간</th><th>분류</th></tr>{rows}</table>',
        unsafe_allow_html=True,
    )
    label_en = {0: "background", 1: "unvoiced", 2: "voiced"}
    report_lines = [f"{seg.start:.2f} ~ {seg.end:.2f} sec: {label_en[seg.code]}" for seg in segments]
    st.download_button(
        "분류 결과 텍스트로 저장",
        data="\n".join(report_lines),
        file_name=f"{Path(clip.name).stem}_segments.txt",
        mime="text/plain",
        width="stretch",
    )
    st.caption(
        "에너지(RMS)가 잡음 바닥보다 충분히 높으면 발화 구간으로, 그중 피치(YIN)가 검출되거나 ZCR이 낮으면 유성음, "
        "그렇지 않으면 무성음으로 분류하는 간단한 규칙 기반 방법입니다. 위 설정을 조정해 경계를 다시 확인해 보세요."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>단시간 분석·합성 (OLA)</h3>', unsafe_allow_html=True)
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        ola_frame_ms = st.slider("프레임 길이 (ms)", 5.0, 50.0, 20.0, 1.0, key="ola_frame_ms")
    with oc2:
        ola_overlap_pct = st.slider("겹침 비율 (%)", 0.0, 90.0, 50.0, 5.0, key="ola_overlap_pct")
    with oc3:
        ola_window = st.selectbox("윈도우 종류", ["hamming", "hann", "boxcar", "blackman"], key="ola_window")
    ola_weighted = st.checkbox("가중치 OLA (윈도우 합으로 정규화해 오차 보정)", value=True, key="ola_weighted")

    fs = frame_signal(clip, frame_ms=ola_frame_ms, overlap_pct=ola_overlap_pct, window=ola_window)
    recon, _ = ola_reconstruct(fs, weighted=ola_weighted)
    profile = window_overlap_profile(ola_window, fs.frame_length, fs.hop_length, n_frames=20)

    st.plotly_chart(plotly_window_sum(profile, clip.sr, fs.hop_length, ola_window), use_container_width=True, config=CHART_CONFIG)
    st.caption(
        f"프레임 {fs.frame_length}샘플({ola_frame_ms:.0f}ms), 이동 {fs.hop_length}샘플({ola_overlap_pct:.0f}% 겹침). "
        "이 선이 평평할수록(값이 일정할수록) 겹쳐서 더했을 때 원신호를 그대로 복원하기 좋다는 뜻입니다 (COLA 조건). "
        "50%나 75% 겹침에서는 hann/hamming 둘 다 평평하지만, 25%처럼 애매한 비율에서는 물결(리플)이 생기는 걸 확인해 보세요."
    )
    st.plotly_chart(plotly_ola_compare(clip.y, recon, clip.sr, "입력 대 재합성 출력 파형"), use_container_width=True, config=CHART_CONFIG)
    st.plotly_chart(plotly_ola_error(clip.y, recon, clip.sr), use_container_width=True, config=CHART_CONFIG)

    n_ola = min(len(clip.y), len(recon))
    ola_err = clip.y[:n_ola].astype(np.float64) - recon[:n_ola].astype(np.float64)
    rmse = float(np.sqrt(np.mean(ola_err ** 2))) if n_ola else 0.0
    input_rms = float(np.sqrt(np.mean(clip.y[:n_ola].astype(np.float64) ** 2))) if n_ola else 0.0
    rel_err_pct = 100 * rmse / input_rms if input_rms > 1e-12 else 0.0
    st.caption(
        f"재구성 오차(RMSE): {rmse:.5f} (입력 신호 크기 대비 {rel_err_pct:.2f}%). "
        f"지금은 {'가중치 OLA 적용' if ola_weighted else '가중치 미적용(단순 덧셈)'} 상태입니다 — "
        "체크박스를 꺼보면 (특히 겹침 비율이 애매할 때) 오차가 커지는 걸 비교할 수 있어요."
    )
    play_y = recon.astype(np.float32)
    peak = float(np.max(np.abs(play_y))) if len(play_y) else 0.0
    if peak > 1.0:
        play_y = play_y / peak
    st.audio(to_wav_bytes(AudioClip(y=play_y, sr=clip.sr, name="ola_recon.wav", channels=1)), format="audio/wav")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>피치 추정: 자기상관 vs AMDF</h3>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pitch_frame_ms = st.slider("프레임 길이 (ms)", 10.0, 50.0, 20.0, 1.0, key="pitch_frame_ms")
    with pc2:
        pitch_overlap_pct = st.slider("겹침 비율 (%)", 0.0, 90.0, 50.0, 5.0, key="pitch_overlap_pct")
    with pc3:
        pitch_conf_th = st.slider("신뢰도 임계값", 0.0, 1.0, 0.3, 0.05, key="pitch_conf_th")
    fix_octave = st.checkbox("옥타브 오류 보정 (피치 더블링/하빙 완화)", value=False, key="fix_octave")

    t_ac, f0_ac, conf_ac = pitch_autocorrelation(clip, frame_ms=pitch_frame_ms, overlap_pct=pitch_overlap_pct)
    t_amdf, f0_amdf, conf_amdf = pitch_amdf(clip, frame_ms=pitch_frame_ms, overlap_pct=pitch_overlap_pct)
    if fix_octave:
        f0_ac = fix_octave_errors(f0_ac, conf_ac)
        f0_amdf = fix_octave_errors(f0_amdf, conf_amdf)

    st.plotly_chart(
        plotly_pitch_compare(t_ac, f0_ac, conf_ac, t_amdf, f0_amdf, conf_amdf, pitch_conf_th),
        use_container_width=True, config=CHART_CONFIG,
    )
    rel_ac = conf_ac >= pitch_conf_th
    rel_amdf = conf_amdf >= pitch_conf_th
    med_ac = f"{np.median(f0_ac[rel_ac]):.0f} Hz" if rel_ac.any() else "—"
    med_amdf = f"{np.median(f0_amdf[rel_amdf]):.0f} Hz" if rel_amdf.any() else "—"
    st.caption(
        f"자기상관: {int(rel_ac.sum())}/{len(conf_ac)} 프레임이 신뢰도 {pitch_conf_th:.2f} 이상 (중앙값 {med_ac}). "
        f"AMDF: {int(rel_amdf.sum())}/{len(conf_amdf)} 프레임 (중앙값 {med_amdf}). "
        "신뢰도 미달 구간은 그래프에서 끊겨서(빈 구간) 표시됩니다."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>VAD (음성 활성 구간 검출)</h3>', unsafe_allow_html=True)
    with st.expander("VAD 기준 설정", expanded=False):
        vad_energy_margin = st.slider("에너지 임계값 여유 (dB, 잡음 바닥 대비)", 3.0, 30.0, 12.0, 1.0, key="vad_energy_margin")
        vad_zcr_th = st.slider("무성음 판정 ZCR 임계값 (이보다 높으면 무성음 후보)", 0.02, 0.4, 0.20, 0.01, key="vad_zcr_th")
        vad_conf_th = st.slider("피치 신뢰도 임계값", 0.0, 1.0, 0.25, 0.05, key="vad_conf_th")
    vad = vad_frames(
        clip, energy_margin_db=vad_energy_margin, zcr_threshold=vad_zcr_th, pitch_conf_threshold=vad_conf_th
    )
    st.plotly_chart(plotly_vad_features(vad), use_container_width=True, config=CHART_CONFIG)
    st.plotly_chart(plotly_vad(clip, vad), use_container_width=True, config=CHART_CONFIG)
    active_ratio = 100 * float(vad["active"].mean()) if len(vad["active"]) else 0.0
    st.caption(
        f"전체 프레임 중 {active_ratio:.0f}%가 음성 활성(active)으로 판정됨. "
        "상대 에너지가 임계값을 넘으면서, 피치가 신뢰도 있게 검출되거나(유성음) ZCR이 높으면(무성음 자음 후보) 활성으로 판정하는 규칙 기반 VAD입니다."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>LPC 분석 (Levinson-Durbin)</h3>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        lpc_frame_ms = st.slider("프레임 길이 (ms)", 10.0, 50.0, 25.0, 1.0, key="lpc_frame_ms")
    with lc2:
        lpc_pos_s = st.slider(
            "분석 위치 (초)", 0.0, duration, min(duration / 3, duration), 0.01, key="lpc_pos_s"
        )
    with lc3:
        lpc_order = st.slider("LPC 차수 (P)", 2, 24, 14, 1, key="lpc_order")

    frame_len = max(int(clip.sr * lpc_frame_ms / 1000), 8)
    if len(clip.y) > frame_len:
        start = max(0, min(int(lpc_pos_s * clip.sr), len(clip.y) - frame_len))
    else:
        start = 0
    frame = clip.y[start:start + frame_len]
    if len(frame) < frame_len:
        frame = np.pad(frame, (0, frame_len - len(frame)))

    lpc_result = lpc_analyze_frame(frame, clip.sr, lpc_order)
    st.plotly_chart(plotly_lpc_signal_error(lpc_result), use_container_width=True, config=CHART_CONFIG)
    st.caption(
        "칠판 수식의 x(n) → [A(z)] → e(n) 과정 그대로입니다: 이 프레임(해밍 윈도우 적용)을 분석 필터 A(z)에 통과시켜 "
        f"잔차 e(n)을 구한 것. 신호 표준편차 {np.std(lpc_result.frame):.4f} → 잔차 표준편차 {np.std(lpc_result.residual):.4f}로 "
        "줄어든 만큼이 '예측 가능했던(=압축으로 아낄 수 있는) 정보'입니다."
    )

    compare_orders = sorted({o for o in (4, 8, 12, lpc_order) if 1 <= o <= 24})[:5]
    st.plotly_chart(
        plotly_lpc_spectrum_compare(frame, clip.sr, compare_orders, lpc_order),
        use_container_width=True, config=CHART_CONFIG,
    )
    st.caption(
        "회색 선은 실제 신호의 FFT 스펙트럼, 색선들은 차수(P)별 LPC(AR) 스펙트럼 — V(ω) = B/A(ω)를 dB로 그린 것입니다. "
        "차수가 높아질수록 포먼트(공명 봉우리)를 더 세밀하게 따라가는 걸 볼 수 있어요."
    )
    with st.expander("반사계수(reflection coefficients) k_i 보기", expanded=False):
        k_rows = "".join(f"<tr><td>k_{i + 1}</td><td>{k:.4f}</td></tr>" for i, k in enumerate(lpc_result.k))
        st.markdown(f'<table class="stats"><tr><th>단계</th><th>값</th></tr>{k_rows}</table>', unsafe_allow_html=True)
        st.caption("Levinson-Durbin 재귀의 각 단계에서 나오는 반사계수입니다. 전부 -1~1 사이에 있어야 필터가 안정적입니다.")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
