"""Audio Lab — speech download, playback, and analysis."""

from __future__ import annotations

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
    download_sample,
    download_url,
    list_downloads,
    load_audio_bytes,
    load_audio_file,
    make_demo,
    synthesize_speech,
    to_wav_bytes,
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

    st.plotly_chart(plotly_waveform(clip, t0, t1), width="stretch", config=CHART_CONFIG)
    st.plotly_chart(plotly_mel(clip, n_fft, hop, n_mels, fmin, fmax, t0, t1, y0, y1), width="stretch", config=CHART_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>스펙트럼 · 음높이</h3>', unsafe_allow_html=True)
    tab_stft, tab_spec, tab_pitch = st.tabs(["선형 스펙트로그램", "평균 스펙트럼", "음높이"])
    with tab_stft:
        st.plotly_chart(plotly_stft(clip, n_fft, hop, t0, t1, y0, y1), width="stretch", config=CHART_CONFIG)
    with tab_spec:
        st.plotly_chart(plotly_spectrum(clip), width="stretch", config=CHART_CONFIG)
    with tab_pitch:
        st.plotly_chart(plotly_pitch(clip, hop), width="stretch", config=CHART_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
