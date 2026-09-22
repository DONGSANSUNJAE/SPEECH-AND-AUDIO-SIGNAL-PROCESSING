"""Build the HW2 (Speech signal analysis) Word report from a recorded WAV file.

Usage:
    .venv/bin/python HW2_SignalAnalysis/generate_report.py [path/to/recording.wav]

If no path is given, picks the most recently saved `downloads/clip_*.wav`
(that's what AudioLab's "이 음성을 downloads/ 폴더에 저장" button writes).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Emu, Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

import audio_core as ac

# Page geometry and heading style copied from the student's own previous
# Classroom submission (Classroom_Assignment_2026311323_Howon_Chung.docx) so
# every homework in this course looks like one consistent document: A4 page,
# Times New Roman, justified body text, manually bolded/sized section titles
# instead of Word's built-in Heading styles.
PAGE_WIDTH = Emu(7560310)
PAGE_HEIGHT = Emu(10692130)
MARGIN_LEFT = Emu(1143000)
MARGIN_RIGHT = Emu(914400)
MARGIN_TOP = Emu(914400)
MARGIN_BOTTOM = Emu(914400)
COURSE_TITLE = "Audio and Signal Processing"

# audio_core sets a dark matplotlib theme on import (for the app's now-unused
# dark-mode plots) — reset to a clean light theme suited for a printed report.
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "font.family": ["Arial Unicode MS", "AppleGothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#1C1916",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#1C1916",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

STUDENT_ID = "2026311323"
STUDENT_NAME = "Howon Chung"
SENTENCE = (
    "Speech has evolved as a primary form of communication between humans. "
    "The topic of this class, ‘discrete-time speech signal processing’ "
    "can be defined as the manipulation of sampled speech signals by a digital "
    "processor to obtain a new signal with some desired properties."
)

SEG_COLORS = {0: "#C9BFAE", 1: "#C97A3B", 2: "#1B3A6B"}
SEG_LABEL_EN = {0: "background", 1: "unvoiced", 2: "voiced"}


def find_recording() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"파일을 찾을 수 없습니다: {p}")
        return p
    candidates = sorted(ac.DOWNLOADS.glob("clip_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(
            "downloads/clip_*.wav 를 찾지 못했습니다. AudioLab 앱에서 '이 음성을 downloads/ 폴더에 저장' 버튼을 먼저 눌러주세요."
        )
    return candidates[0]


def fig_to_stream(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_waveform_fig(clip: ac.AudioClip, title: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    times = np.arange(len(clip.y)) / clip.sr
    ax.plot(times, clip.y, color="#1B3A6B", linewidth=0.6)
    ax.set_xlim(0, max(clip.duration, 0.05))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    return fig_to_stream(fig)


def make_spectrum_overlay_fig(entries: list[tuple[str, ac.AudioClip, str]]) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    max_sr = max(clip.sr for _, clip, _ in entries)
    for label, clip, color in entries:
        n = len(clip.y)
        mag = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(clip.y * np.hanning(n))), 1e-12))
        mag -= np.max(mag)
        freqs = np.fft.rfftfreq(n, d=1 / clip.sr)
        ax.plot(freqs, mag, label=label, color=color, linewidth=1.0)
    ax.set_xlim(0, max_sr / 2)
    ax.set_ylim(-80, 3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Spectrum comparison (overlaid)", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig_to_stream(fig)


def make_spectrogram_grid_fig(entries: list[tuple[str, ac.AudioClip]], freq_range: float) -> io.BytesIO:
    fig, axes = plt.subplots(1, len(entries), figsize=(7.2, 2.6), sharey=True)
    if len(entries) == 1:
        axes = [axes]
    n_fft, hop = 1024, 128
    for ax, (label, clip) in zip(axes, entries):
        S = ac.librosa.amplitude_to_db(np.abs(ac.librosa.stft(clip.y, n_fft=n_fft, hop_length=hop)), ref=np.max)
        img = ac.librosa.display.specshow(
            S, sr=clip.sr, hop_length=hop, x_axis="time", y_axis="hz", cmap="inferno", ax=ax, vmin=-80, vmax=0
        )
        ax.set_ylim(0, freq_range)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=axes, format="%+d dB", fraction=0.03, pad=0.02)
    fig.suptitle(
        f"Spectrograms on a fixed 0–{freq_range:,.0f} Hz axis and color scale (for fair comparison)",
        fontsize=10, y=1.05,
    )
    return fig_to_stream(fig)


def make_segments_fig(clip: ac.AudioClip, segments) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    for seg in segments:
        ax.axvspan(seg.start, seg.end, color=SEG_COLORS.get(seg.code, "#CCCCCC"), alpha=0.35, lw=0)
    times = np.arange(len(clip.y)) / clip.sr
    ax.plot(times, clip.y, color="black", linewidth=0.6)
    ax.set_xlim(0, max(clip.duration, 0.05))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform with automatic voiced(navy)/unvoiced(orange)/background(tan) segments", fontsize=10, loc="left")
    fig.tight_layout()
    return fig_to_stream(fig)


def make_energy_zcr_fig(times: np.ndarray, diag: dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(times, diag["rms_db"], color="#1B3A6B", linewidth=1.1, label="Energy (dB)")
    ax.axhline(diag["threshold_db"], color="#A8481C", linestyle="--", linewidth=1.0,
               label=f"threshold {diag['threshold_db']:.1f} dB")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (dB)")
    ax2 = ax.twinx()
    ax2.plot(times, diag["zcr"], color="#C97A3B", linewidth=0.8, label="ZCR")
    ax2.set_ylabel("ZCR")
    ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.set_title("Frame-level energy (RMS) and zero-crossing rate", fontsize=11, loc="left")
    fig.tight_layout()
    return fig_to_stream(fig)


def add_picture(doc: Document, stream: io.BytesIO, width_in: float = 6.0) -> None:
    doc.add_picture(stream, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def set_page_geometry(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = PAGE_WIDTH
    section.page_height = PAGE_HEIGHT
    section.left_margin = MARGIN_LEFT
    section.right_margin = MARGIN_RIGHT
    section.top_margin = MARGIN_TOP
    section.bottom_margin = MARGIN_BOTTOM


def add_title(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(17)


def add_id_line(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(1.5)


def add_section(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)


def add_body(doc: Document, text: str, italic: bool = False) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.italic = italic


def build_report(clip: ac.AudioClip, out_path: Path, native_sr: int | None = None) -> Path:
    targets = [t for t in (16000, 8000, 2000) if t < clip.sr] or [clip.sr]
    downs = {t: ac.resample_clip(clip, t) for t in targets}
    compare_colors = {16000: "#1B3A6B", 8000: "#C97A3B", 2000: "#A8481C"}

    times_c, _, segments, diag = ac.classify_segments(clip)

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    set_page_geometry(doc)

    add_title(doc, COURSE_TITLE)
    add_id_line(doc, f"{STUDENT_ID} {STUDENT_NAME}")
    doc.add_paragraph()

    add_section(doc, "1. Introduction")
    add_body(
        doc,
        "This report records a spoken sentence, compares it after downsampling to 16 kHz, 8 kHz and 2 kHz, "
        "and analyzes it by classifying each frame as background, unvoiced, or voiced speech with timing "
        "information. It closes with ideas for automating that segmentation."
    )

    add_section(doc, "2. Recording")
    if native_sr and native_sr != clip.sr:
        add_body(
            doc,
            f"The sentence below was recorded through the browser microphone, which captured audio "
            f"natively at {native_sr:,} Hz, mono. To meet this assignment's SF = {clip.sr:,} Hz requirement, "
            f"the recording was resampled to {clip.sr:,} Hz mono for submission. Note that resampling upward "
            f"does not add back any frequency content above the original capture's "
            f"{native_sr / 2:,.0f} Hz Nyquist limit — it only conforms the file to the required sample rate."
        )
    else:
        add_body(doc, f"The sentence below was recorded at {clip.sr:,} Hz, mono.")
    add_body(doc, f"“{SENTENCE}”", italic=True)
    add_picture(doc, make_waveform_fig(clip, f"Recorded waveform ({clip.sr:,} Hz, {clip.duration:.2f} s)"))

    add_section(doc, "3. Down-sampling Comparison (16 / 8 / 2 kHz)")
    add_body(
        doc,
        "Down-sampling by an integer factor lowers the sampling rate, and with it the Nyquist frequency "
        "fs/2 — the highest frequency the signal can represent. Any content above the new Nyquist frequency "
        "is removed (after anti-alias filtering) rather than folded back, so a lower sampling rate simply "
        "means a narrower retained bandwidth. The recording was down-sampled to 16 kHz, 8 kHz, and 2 kHz "
        "to observe this directly."
    )
    add_body(
        doc,
        "The overlaid spectrum below shows each version's content being cut off at its own Nyquist "
        "frequency; the spectrograms alongside share an identical frequency axis and color scale (0 Hz to "
        "the original Nyquist) so the shrinking retained bandwidth is directly comparable across the three "
        "versions."
    )
    entries_spec = [("Original", clip, "#6A635B")] + [
        (f"{t:,} Hz", downs[t], compare_colors.get(t, "#333333")) for t in targets
    ]
    add_picture(doc, make_spectrum_overlay_fig(entries_spec))
    add_picture(doc, make_spectrogram_grid_fig([(f"{t:,} Hz", downs[t]) for t in targets], freq_range=clip.sr / 2))
    notes = {
        16000: "Retains content up to 8 kHz. Most consonant/fricative cues survive; hard to tell apart from the original by ear.",
        8000: "Retains content only up to 4 kHz — telephone-quality audio; sibilants (s, sh) sound noticeably duller.",
        2000: "Retains content only up to 1 kHz — most fricative/stop cues are lost. Vowels remain audible but consonants become hard to distinguish.",
    }
    for t in targets:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(f"{t:,} Hz: ")
        r.bold = True
        p.add_run(notes.get(t, ""))

    add_section(doc, "4. Signal Analysis: Voiced / Unvoiced / Background Classification")
    add_body(
        doc,
        "Each frame's short-time energy (RMS, in dB) and zero-crossing rate (ZCR) were computed using "
        "25 ms frames with a 10 ms hop (60% overlap). A frame is classified as background(0) if its energy "
        "is near the estimated noise floor; otherwise it is voiced(2) if a reliable pitch (YIN) is detected "
        "or ZCR is low, and unvoiced(1) otherwise."
    )
    add_picture(doc, make_energy_zcr_fig(times_c, diag))
    add_picture(doc, make_segments_fig(clip, segments))

    add_body(doc, "Classified regions and timing:")
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text = "Time range", "Class"
    for seg in segments:
        row = table.add_row().cells
        row[0].text = f"{seg.start:.2f} ~ {seg.end:.2f} sec"
        row[1].text = SEG_LABEL_EN[seg.code]

    doc.add_paragraph()
    add_section(doc, "5. Automatic Segmentation")
    add_body(doc, "Ideas for automating the region classification performed manually above:")
    ideas = [
        ("Short-time energy + ZCR thresholding",
         "The method used above: split background/speech by per-frame RMS energy, then split voiced/unvoiced "
         "by ZCR. Simple to implement, but thresholds must be re-tuned for every noise condition."),
        ("Adaptive thresholding",
         "Estimate the noise floor from the recording's own quiet frames (as done here) so the energy "
         "threshold sets itself automatically. Re-estimating it in a sliding window handles noise levels "
         "that drift over the course of a recording."),
        ("Periodicity-based voiced/unvoiced decision",
         "Use autocorrelation or the cepstrum: a clear peak indicates vocal-fold vibration (voiced), no peak "
         "indicates unvoiced. More robust to noise than ZCR alone."),
        ("Spectral features",
         "Spectral flux (frame-to-frame spectral change) can detect segment boundaries (onsets); spectral "
         "entropy or the low/high-frequency energy ratio can reinforce fricative detection."),
        ("Statistical / model-based methods",
         "Train a GMM classifier on features such as (energy, ZCR, pitch strength, spectral tilt), or decode "
         "a 3-state (silence/unvoiced/voiced) HMM à la Rabiner–Sambur for a temporally smooth label "
         "sequence with less frame-to-frame flicker than a rule-based decision."),
        ("Deep-learning VAD/segmentation",
         "Use a pretrained voice-activity detector (WebRTC VAD, Silero VAD) or train a frame-level CNN/RNN "
         "classifier on labeled data. More robust to noise and channel variation than hand-tuned rules."),
    ]
    for name, desc in ideas:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(f"{name}: ")
        r.bold = True
        p.add_run(desc)

    doc.save(out_path)
    return out_path


REQUIRED_SR = 48000


def main() -> None:
    wav_path = find_recording()
    clip = ac.load_audio_file(wav_path)
    native_sr = clip.sr
    if clip.sr != REQUIRED_SR:
        clip = ac.resample_clip(clip, REQUIRED_SR)
    out_path = ROOT / "HW2_SignalAnalysis" / f"Homework2_{STUDENT_ID}_{STUDENT_NAME.replace(' ', '')}.docx"
    build_report(clip, out_path, native_sr=native_sr)
    print(f"Wrote {out_path} (source: {wav_path.name}, native {native_sr} Hz -> {clip.sr} Hz, {clip.duration:.2f}s)")


if __name__ == "__main__":
    main()
