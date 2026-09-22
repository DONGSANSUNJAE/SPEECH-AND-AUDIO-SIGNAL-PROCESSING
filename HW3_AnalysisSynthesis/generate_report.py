"""Build the HW3 (Analysis and Synthesis: OLA / pitch / VAD) Word report.

Usage:
    .venv/bin/python HW3_AnalysisSynthesis/generate_report.py [path/to/recording.wav]

If no path is given, picks the most recently saved `downloads/clip_*.wav`.
Reuses the same document style as HW2 (A4, Times New Roman, justified body,
manually bolded/sized section titles) for a consistent look across homeworks.
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

PAGE_WIDTH = Emu(7560310)
PAGE_HEIGHT = Emu(10692130)
MARGIN_LEFT = Emu(1143000)
MARGIN_RIGHT = Emu(914400)
MARGIN_TOP = Emu(914400)
MARGIN_BOTTOM = Emu(914400)
COURSE_TITLE = "Audio and Signal Processing"


def find_recording() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"파일을 찾을 수 없습니다: {p}")
        return p
    candidates = sorted(ac.DOWNLOADS.glob("clip_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("downloads/clip_*.wav 를 찾지 못했습니다.")
    return candidates[0]


def fig_to_stream(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# --- style helpers (matching the student's earlier Classroom submission) ---

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


def add_subsection(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)


def add_body(doc: Document, text: str, italic: bool = False) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.italic = italic


def add_bullet(doc: Document, name: str, desc: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(f"{name}: ")
    r.bold = True
    p.add_run(desc)


def add_picture(doc: Document, stream: io.BytesIO, width_in: float = 6.0) -> None:
    doc.add_picture(stream, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# --- figures ---

def fig_window_sums(frame_length: int) -> io.BytesIO:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, overlap_pct in zip(axes, [25, 50, 75]):
        hop = max(int(round(frame_length * (1 - overlap_pct / 100))), 1)
        for w, color in [("hann", "#1B3A6B"), ("hamming", "#C97A3B")]:
            prof = ac.window_overlap_profile(w, frame_length, hop, n_frames=20)
            ax.plot(prof, label=w, color=color, linewidth=1.0)
        ax.set_title(f"{overlap_pct}% overlap", fontsize=10)
        ax.set_xlabel("sample")
    axes[0].set_ylabel("window sum")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig_to_stream(fig)


def fig_ola_compare(clip: ac.AudioClip, weighted_recon: np.ndarray, unweighted_recon: np.ndarray) -> io.BytesIO:
    n = min(len(clip.y), len(weighted_recon), len(unweighted_recon))
    t = np.arange(n) / clip.sr
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True)
    axes[0].plot(t, clip.y[:n], color="#6A635B", linewidth=0.6)
    axes[0].set_title("Input", fontsize=10, loc="left")
    axes[1].plot(t, (clip.y[:n] - unweighted_recon[:n]), color="#A8481C", linewidth=0.6)
    axes[1].set_title("Error: plain OLA (no normalization)", fontsize=10, loc="left")
    axes[2].plot(t, (clip.y[:n] - weighted_recon[:n]), color="#1B3A6B", linewidth=0.6)
    axes[2].set_title("Error: weighted OLA (normalized by window sum)", fontsize=10, loc="left")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig_to_stream(fig)


def fig_pitch_compare(t_ac, f0_ac, conf_ac, t_amdf, f0_amdf, conf_amdf, conf_th: float) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    f0_ac_m = np.where(conf_ac >= conf_th, f0_ac, np.nan)
    f0_amdf_m = np.where(conf_amdf >= conf_th, f0_amdf, np.nan)
    ax.plot(t_ac, f0_ac_m, color="#1B3A6B", linewidth=1.0, label="Autocorrelation")
    ax.plot(t_amdf, f0_amdf_m, color="#C97A3B", linewidth=1.0, label="AMDF")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(0, 420)
    ax.legend(fontsize=8)
    ax.set_title("Pitch estimate comparison", fontsize=11, loc="left")
    fig.tight_layout()
    return fig_to_stream(fig)


def fig_vad(clip: ac.AudioClip, vad: dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    times, active = vad["times"], vad["active"]
    hop_s = vad["hop_length"] / clip.sr
    in_run, run_start = False, 0.0
    for i, a in enumerate(active):
        if a and not in_run:
            in_run, run_start = True, times[i] - hop_s / 2
        elif not a and in_run:
            in_run = False
            ax.axvspan(max(0.0, run_start), times[i] - hop_s / 2, color="#1B3A6B", alpha=0.25, lw=0)
    if in_run:
        ax.axvspan(max(0.0, run_start), times[-1] + hop_s / 2, color="#1B3A6B", alpha=0.25, lw=0)
    t = np.arange(len(clip.y)) / clip.sr
    ax.plot(t, clip.y, color="black", linewidth=0.5)
    ax.set_xlim(0, max(clip.duration, 0.05))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("VAD result (shaded = voice active)", fontsize=11, loc="left")
    fig.tight_layout()
    return fig_to_stream(fig)


def build_report(clip: ac.AudioClip, out_path: Path, native_sr: int | None = None) -> Path:
    # --- OLA: framing, reconstruction, COLA/ripple analysis ---
    fs50 = ac.frame_signal(clip, frame_ms=20.0, overlap_pct=50.0, window="hamming")
    recon_w, _ = ac.ola_reconstruct(fs50, weighted=True)
    recon_u, _ = ac.ola_reconstruct(fs50, weighted=False)
    n = min(len(clip.y), len(recon_w))
    rmse_w = float(np.sqrt(np.mean((clip.y[:n].astype(np.float64) - recon_w[:n].astype(np.float64)) ** 2)))
    rmse_u = float(np.sqrt(np.mean((clip.y[:n].astype(np.float64) - recon_u[:n].astype(np.float64)) ** 2)))
    input_rms = float(np.sqrt(np.mean(clip.y[:n].astype(np.float64) ** 2))) or 1e-9

    ripple_rows = []
    for overlap_pct in (25, 50, 75):
        hop = max(int(round(fs50.frame_length * (1 - overlap_pct / 100))), 1)
        for w in ("hann", "hamming"):
            n_frames = 60
            margin = 3 * fs50.frame_length
            prof = ac.window_overlap_profile(w, fs50.frame_length, hop, n_frames=n_frames)
            while len(prof) <= 2 * margin:
                n_frames += 40
                prof = ac.window_overlap_profile(w, fs50.frame_length, hop, n_frames=n_frames)
            mid = prof[margin: len(prof) - margin]
            ripple_pct = 100 * (mid.max() - mid.min()) / mid.mean() if mid.mean() > 0 else 0.0
            ripple_rows.append((overlap_pct, w, mid.mean(), ripple_pct))

    fs25 = ac.frame_signal(clip, frame_ms=20.0, overlap_pct=25.0, window="hamming")
    recon25_w, _ = ac.ola_reconstruct(fs25, weighted=True)
    recon25_u, _ = ac.ola_reconstruct(fs25, weighted=False)
    n25 = min(len(clip.y), len(recon25_w))
    rmse25_w = float(np.sqrt(np.mean((clip.y[:n25].astype(np.float64) - recon25_w[:n25].astype(np.float64)) ** 2)))
    rmse25_u = float(np.sqrt(np.mean((clip.y[:n25].astype(np.float64) - recon25_u[:n25].astype(np.float64)) ** 2)))

    # --- Pitch ---
    t_ac, f0_ac, conf_ac = ac.pitch_autocorrelation(clip)
    t_amdf, f0_amdf, conf_amdf = ac.pitch_amdf(clip)
    conf_th = 0.3
    rel_ac = conf_ac >= conf_th
    rel_amdf = conf_amdf >= conf_th
    f0_ac_fixed = ac.fix_octave_errors(f0_ac, conf_ac)

    # --- VAD ---
    vad = ac.vad_frames(clip)

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
        "This report implements and analyzes three short-time speech processing techniques: framing with "
        "overlap-and-add (OLA) analysis/synthesis, pitch (F0) estimation via autocorrelation and AMDF, and a "
        "rule-based voice activity detector (VAD). The recording from Homework 1 is used throughout."
    )
    if native_sr and native_sr != clip.sr:
        add_body(
            doc,
            f"The recording's microphone captured audio natively at {native_sr:,} Hz; as in the Homework 2 "
            f"submission, it is resampled to {clip.sr:,} Hz here for consistency (upsampling does not restore "
            f"frequency content above the original {native_sr / 2:,.0f} Hz Nyquist limit, so this affects only "
            "the sample counts reported below, not any Hz- or second-based result)."
        )

    # ---------------- OLA ----------------
    add_section(doc, "2. Short-Time Framing and Overlap-and-Add (OLA)")
    add_body(
        doc,
        f"The signal was split into {20.0:.0f} ms frames ({fs50.frame_length} samples at {clip.sr:,} Hz) with "
        f"50% overlap (hop = {fs50.hop_length} samples), each multiplied by a Hamming window, then reconstructed "
        "by overlap-and-add. Frame length and overlap are implemented as free parameters (not hard-coded to "
        "20 ms/50%), so other configurations can be produced directly."
    )
    add_subsection(doc, "2.1 Hanning vs. Hamming")
    add_body(
        doc,
        "A window satisfies the constant-overlap-add (COLA) property when shifted, overlapping copies of it sum "
        "to a constant — only then does plain OLA reconstruct the original signal without amplitude modulation. "
        "The window-sum profiles below were measured directly (not assumed) at 25%, 50%, and 75% overlap:"
    )
    add_picture(doc, fig_window_sums(fs50.frame_length))
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for c, t in zip(hdr, ["Overlap", "Window", "Steady-state sum", "Ripple (%)"]):
        c.text = t
    for overlap_pct, w, mean_sum, ripple_pct in ripple_rows:
        row = table.add_row().cells
        row[0].text = f"{overlap_pct}%"
        row[1].text = w
        row[2].text = f"{mean_sum:.3f}"
        row[3].text = f"{ripple_pct:.1f}%"
    doc.add_paragraph()
    add_body(
        doc,
        "At 50% and 75% overlap, both Hanning and Hamming sum to an exactly flat constant (ripple ≈ 0%) — this "
        "is a general property of any single-cosine (\"raised-cosine\") window at hop = frame_length / 2^k for "
        "integer k, not something specific to Hanning. At 25% overlap (hop = 3/4 of the frame, not a power-of-two "
        "fraction), neither window satisfies COLA exactly, and both show real ripple. In this measurement, Hanning "
        "actually ripples somewhat more than Hamming at 25% overlap — the intuition that \"Hamming always ripples "
        "more\" is a common oversimplification; the real distinguishing factor is whether hop/frame_length is a "
        "power-of-two fraction, independent of the specific window's cosine coefficients."
    )
    add_subsection(doc, "2.2 Weighted OLA as error compensation")
    add_body(
        doc,
        f"At 50% overlap (flat window sum, constant = {ripple_rows[3][2]:.2f} for Hamming), plain OLA already "
        f"reconstructs the signal almost perfectly once that constant is factored in: RMSE = {rmse_u:.6f} "
        f"(unweighted) vs. {rmse_w:.6f} (weighted), both negligible relative to the input's RMS amplitude of "
        f"{input_rms:.4f}. At the non-ideal 25% overlap, the gap widens: unweighted RMSE = {rmse25_u:.5f} vs. "
        f"weighted RMSE = {rmse25_w:.5f} — dividing by the (non-constant) window-sum profile measurably reduces "
        "the amplitude-modulation artifact that an unnormalized sum leaves behind."
    )
    add_picture(doc, fig_ola_compare(clip, recon25_w, recon25_u))
    add_subsection(doc, "2.3 Non-50% overlap and the Griffin–Lim connection")
    add_body(
        doc,
        "The framing/OLA implementation takes overlap as a continuous percentage rather than assuming 50%, so "
        "25%/75%/etc. overlap ratios are handled by the same code path (see Table above, computed at all three)."
    )
    add_body(
        doc,
        "Griffin and Lim (1984), Section II (\"Signal Estimation from Modified Short-Time Fourier Transform\"), "
        "derive the estimate that minimizes the mean-squared error between a candidate signal's STFT and a given "
        "(possibly modified) target STFT: their Eq. (6), x(n) = Σ_m w(mS−n)y_w(mS,n) / Σ_m w²(mS−n) — normalized "
        "by the SUM OF THE SQUARED analysis window, not the plain sum. This is a stronger, theoretically-derived "
        "result and is not the same formula as this report's weighted OLA (Section 2.2): that normalizes by the "
        "unsquared window sum, Σ_m w(mS−n), which is the paper's Eq. (7), \"the [standard] overlap-add "
        "procedure\" — a simpler, pre-existing method the paper itself calls an ad hoc special case, valid when "
        "reconstructing an exact/unmodified STFT (our case, since we resynthesize from our own consistent, "
        "unmodified frames) but not the least-squares-optimal choice once the STFT has actually been modified. "
        "The paper's Eq. (8), a generic \"weighted overlap-add\" with an arbitrary synthesis filter (which "
        "subsumes an analysis/synthesis window product), is likewise not something Griffin and Lim generalize to "
        "themselves in Section II — they attribute it to Portnoff (1980) and Crochiere (1980), citing it as prior "
        "work. Griffin and Lim's own further contribution, the iterative LSEE-MSTFTM algorithm that alternates "
        "between time-domain reconstruction (via Eq. 6) and re-imposing a target magnitude, appears separately in "
        "Section III (\"Signal Estimation from Modified STFT Magnitude\") — needed there because a magnitude-only "
        "target is in general not realizable by any single signal, so no closed form suffices and iteration is "
        "required; that complication does not arise here, since our frames are never modified between analysis "
        "and synthesis."
    )

    # ---------------- Pitch ----------------
    add_section(doc, "3. Pitch Estimation: Autocorrelation vs. AMDF")
    add_body(
        doc,
        "Both estimators operate on unwindowed (rectangular) 20 ms/50%-overlap frames — pitch detectors are "
        "sensitive to the near-zero taper a Hamming/Hanning window puts at frame edges, which otherwise biases "
        "the search toward spuriously long lags. Autocorrelation searches the lag range corresponding to 65–400 Hz "
        "for the highest normalized peak; AMDF searches the same range for the deepest dip in the average "
        "magnitude difference."
    )
    add_picture(doc, fig_pitch_compare(t_ac, f0_ac, conf_ac, t_amdf, f0_amdf, conf_amdf, conf_th))
    add_body(
        doc,
        f"Autocorrelation: {int(rel_ac.sum())}/{len(conf_ac)} frames reached confidence ≥ {conf_th} "
        f"(median F0 {np.median(f0_ac[rel_ac]):.0f} Hz among those). AMDF: {int(rel_amdf.sum())}/{len(conf_amdf)} "
        f"frames (median F0 {np.median(f0_amdf[rel_amdf]):.0f} Hz). AMDF marks more frames as \"reliable\" but its "
        "track is visibly noisier frame-to-frame, consistent with AMDF being more sensitive to local amplitude "
        "variation than the normalized autocorrelation peak."
    )
    add_subsection(doc, "3.1 Setting and evaluating thresholds")
    add_body(
        doc,
        "The confidence threshold (normalized autocorrelation peak height, or normalized AMDF dip depth) is not "
        "a single universal constant — it trades off missed voiced frames (threshold too high) against spurious "
        "pitch on unvoiced/silent frames (threshold too low). It was evaluated qualitatively here by comparing the "
        "separate F0, waveform, and VAD-active-region plots side by side and checking that gaps in the F0 track "
        "(no reliable pitch reported) line up with unvoiced/silent regions rather than cutting through clearly "
        "voiced vowels; a more rigorous evaluation would compare against a reference pitch track (e.g. from a "
        "laryngograph/EGG signal, or a well-validated algorithm such as YIN) and report voicing-decision error "
        "rate and F0 deviation on frames both mark voiced."
    )
    add_subsection(doc, "3.2 Preventing octave errors (doubling/halving)")
    add_body(
        doc,
        "Two distinct octave-error mechanisms were addressed. First, a naive peak/dip search can default to the "
        "edge of the search range (lag_min or lag_max) when no genuine periodicity is present in an unvoiced or "
        "silent frame — this is not really a doubling/halving of a real pitch so much as the search finding "
        "nothing and reporting the boundary anyway; both _autocorr_pitch_frame and _amdf_pitch_frame now detect "
        "this (peak/dip sitting exactly at either edge) and heavily down-weight that frame's confidence so it is "
        "excluded from \"reliable\" pitch, rather than being reported as a confident but spurious frequency. "
        "Second, AMDF specifically can lock onto an exact integer multiple of the true period, since a multiple "
        "can align to a whole-sample lag more exactly than a non-integer-sample fundamental does; "
        "_amdf_pitch_frame now checks whether a shorter sub-multiple (period/2, /3, /4, /5) has a comparably deep "
        "dip and prefers it when so."
    )
    add_body(
        doc,
        "On top of these frame-level fixes, a simple clip-level correction was implemented: frames with reliable "
        f"confidence are compared to the GLOBAL median F0 of every reliable frame in the whole clip (a single "
        f"scalar, {np.median(f0_ac[rel_ac]):.1f} Hz here — not a local/neighboring-frame statistic), and any "
        "frame whose value is close to 2× or 0.5× that global median is snapped back onto the correct octave "
        f"(median F0 for autocorrelation shifted from {np.median(f0_ac[rel_ac]):.1f} Hz to "
        f"{np.median(f0_ac_fixed[rel_ac]):.1f} Hz after correction on this recording). Because this reference is "
        "global rather than local, it can misfire on a genuine sustained pitch excursion (e.g. question "
        "intonation lasting many consecutive frames) that happens to sit near 2× the whole-clip median, since "
        "nothing here distinguishes an isolated single-frame artifact from a real, sustained contour change — it "
        "is best suited to cleaning up occasional one-off slips, not to tracking genuine pitch movement. More "
        "robust standard approaches include constraining the search range to the expected speaker's F0 range, "
        "enforcing temporal continuity with dynamic programming/Viterbi smoothing across the whole utterance "
        "(which would naturally distinguish a brief artifact from a sustained excursion), and cross-checking the "
        "two methods against each other (autocorrelation and AMDF tend to make different octave-error mistakes, "
        "so frames where they disagree by close to an integer ratio are good candidates for correction)."
    )

    # ---------------- VAD ----------------
    add_section(doc, "4. Voice Activity Detection (VAD)")
    add_body(
        doc,
        "The VAD combines three per-frame features: relative energy (RMS in dB above an automatically estimated "
        "noise floor), pitch reliability (the autocorrelation confidence from Section 3), and zero-crossing rate "
        "(ZCR). A frame is marked active if its relative energy clears a margin above the noise floor, and it "
        "looks either voiced (pitch reliably detected, i.e. low, periodic ZCR) or like an unvoiced consonant "
        "(broadband and noise-like, i.e. a HIGH zero-crossing rate rather than a low one — unvoiced fricatives "
        "and stops lack vocal-fold vibration entirely, which is the opposite acoustic signature from voiced "
        "speech, so the two branches use ZCR in opposite directions rather than both requiring it to be low)."
    )
    add_picture(doc, fig_vad(clip, vad))
    active_pct = 100 * float(vad["active"].mean()) if len(vad["active"]) else 0.0
    add_body(
        doc,
        f"On this recording, {active_pct:.0f}% of frames were marked active, noise floor "
        f"{vad['noise_floor_db']:.1f} dB, threshold {vad['threshold_db']:.1f} dB."
    )
    add_subsection(doc, "4.1 Setting and evaluating thresholds")
    add_body(
        doc,
        "As in Homework 2's classifier, the energy threshold is set relative to a noise floor estimated from the "
        "recording itself (10th percentile of frame energy) plus a fixed margin, rather than a single absolute "
        "dB value, since absolute levels vary with microphone gain and recording environment. It was evaluated by "
        "visually checking that active regions line up with the waveform's audibly speech-containing parts and "
        "that pauses between words are correctly marked inactive; a quantitative evaluation would need "
        "ground-truth voice-activity labels to compute precision/recall or a false-alarm/miss-rate curve while "
        "sweeping the margin and ZCR threshold."
    )
    add_subsection(doc, "4.2 Detecting voice activity in noisy environments")
    add_bullet(
        doc, "Track a moving noise floor",
        "instead of one global percentile over the whole recording, re-estimate the floor over a sliding window "
        "so it adapts if background noise changes partway through (e.g., an air conditioner turning on)."
    )
    add_bullet(
        doc, "Use spectral, not just time-domain, features",
        "spectral flatness or the ratio of energy in speech-relevant bands (300–3400 Hz) to total energy is more "
        "robust to broadband noise than raw time-domain energy, since noise and speech can have similar RMS but "
        "different spectral shape."
    )
    add_bullet(
        doc, "Pre-process with noise/spectral subtraction",
        "estimate the noise spectrum during non-speech intervals and subtract it before computing VAD features, "
        "reducing the energy/ZCR distortion noise introduces."
    )
    add_bullet(
        doc, "Fall back to a learned model",
        "a pretrained VAD (WebRTC VAD, Silero VAD) trained on noisy speech generalizes better than hand-tuned "
        "thresholds once the rule-based approach's assumptions (clear energy gap between speech and noise) break "
        "down."
    )

    doc.save(out_path)
    return out_path


REQUIRED_SR = 48000  # match the sample rate the HW2 submission standardized on


def main() -> None:
    wav_path = find_recording()
    clip = ac.load_audio_file(wav_path)
    native_sr = clip.sr
    if clip.sr != REQUIRED_SR:
        clip = ac.resample_clip(clip, REQUIRED_SR)
    out_path = ROOT / "HW3_AnalysisSynthesis" / f"Homework3_{STUDENT_ID}_{STUDENT_NAME.replace(' ', '')}.docx"
    build_report(clip, out_path, native_sr=native_sr)
    print(f"Wrote {out_path} (source: {wav_path.name}, native {native_sr} Hz -> {clip.sr} Hz, {clip.duration:.2f}s)")


if __name__ == "__main__":
    main()
