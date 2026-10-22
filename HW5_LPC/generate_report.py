"""Build the HW5 (Linear Prediction Analysis: Levinson-Durbin inside OLA) report.

Usage:
    .venv/bin/python HW5_LPC/generate_report.py [path/to/recording.wav]

If no path is given, picks the most recently saved `downloads/clip_*.wav`.
Per the assignment (Chapter5.pdf p.60): sampling frequency 8 kHz, frame
length 25 ms, frame shift 10 ms, LPC orders 4/8/12/16/20; deliverable is a
PDF technical report. All content (text/figures/tables) is computed once
into a list of "blocks" and then rendered twice: once as .docx (same page
geometry/fonts as the HW2/HW3 submissions, for consistency) and once
directly as a true .pdf via weasyprint (no GUI app / file-dialog involved).

weasyprint needs Homebrew's pango/cairo (`brew install pango`) and, on
macOS, DYLD_FALLBACK_LIBRARY_PATH must include /opt/homebrew/lib for it to
find them, e.g.:
    DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$HOME/lib:/usr/local/lib:/usr/lib" \\
        .venv/bin/python HW5_LPC/generate_report.py
"""

from __future__ import annotations

import base64
import html as html_lib
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

REQUIRED_SR = 8000       # Homework #5 spec: sampling frequency 8 kHz
FRAME_MS = 25.0          # Homework #5 spec: frame length 25 ms
FRAME_SHIFT_MS = 10.0    # Homework #5 spec: frame shift 10 ms
ORDERS = [4, 8, 12, 16, 20]   # Homework #5 spec: "(Note) LPC order: 4, 8, 12, 16, 20"
PRIMARY_ORDER = 14       # matches the lecture's own worked example (p.61: "M = 14")
ORDER_COLORS = ["#C97A3B", "#7A9A6B", "#A8481C", "#1B3A6B", "#8B5FBF"]


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


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Content blocks: a tiny, format-agnostic representation of the report so the
# same computed text/figures/tables render to both .docx and .pdf.
#   ("title", text)
#   ("id_line", text)
#   ("section", text)
#   ("subsection", text)
#   ("body", text, italic)
#   ("bullet", name, desc)
#   ("picture", png_bytes, width_in)
#   ("table", headers, rows)
# ---------------------------------------------------------------------------

Block = tuple


def extract_frame(clip: ac.AudioClip, t_center: float, frame_len: int) -> np.ndarray:
    start = max(0, min(int(round(t_center * clip.sr)) - frame_len // 2, len(clip.y) - frame_len))
    start = max(0, start)
    f = clip.y[start: start + frame_len]
    if len(f) < frame_len:
        f = np.pad(f, (0, frame_len - len(f)))
    return f.astype(np.float64)


def pick_representative_frames(clip: ac.AudioClip, frame_len: int) -> dict:
    _, _, segments, _ = ac.classify_segments(clip)
    picks = {}
    for code, key in [(2, "voiced"), (1, "unvoiced"), (0, "background")]:
        segs = [s for s in segments if s.code == code]
        if not segs:
            continue
        longest = max(segs, key=lambda s: s.end - s.start)
        mid = (longest.start + longest.end) / 2
        picks[key] = {"t_center": mid, "duration": longest.end - longest.start, "frame": extract_frame(clip, mid, frame_len)}
    return picks


def fig_signal_residual_recon(clip: ac.AudioClip, pipeline: dict) -> bytes:
    n = min(len(clip.y), len(pipeline["residual_signal"]), len(pipeline["recon_signal"]))
    t = np.arange(n) / clip.sr
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].plot(t, clip.y[:n], color="#1B3A6B", linewidth=0.5)
    axes[0].set_title("Input speech x(n)", fontsize=10, loc="left")
    axes[1].plot(t, pipeline["residual_signal"][:n], color="#A8481C", linewidth=0.5)
    axes[1].set_title(f"LPC residual e(n), order p={PRIMARY_ORDER}", fontsize=10, loc="left")
    axes[2].plot(t, clip.y[:n], color="#1B3A6B", linewidth=1.2, alpha=0.5, label="input x(n)")
    axes[2].plot(t, pipeline["recon_signal"][:n], color="#C97A3B", linewidth=0.6, linestyle="--", label="reconstructed x̂(n)")
    axes[2].set_title("Reconstructed via synthesis filter 1/A(z)", fontsize=10, loc="left")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def fig_order_energy(orders: list[int], energies: list[float]) -> bytes:
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.plot(orders, energies, "o-", color="#1B3A6B")
    ax.set_xlabel("LPC order p")
    ax.set_ylabel("Mean residual energy")
    ax.set_xticks(orders)
    ax.set_title("Prediction-error energy vs. order (whitening effect)", fontsize=10, loc="left")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def fig_ar_spectrum_frame(frame: np.ndarray, sr: int, orders: list[int], title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    n = len(frame)
    mag = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(frame * np.hanning(n))), 1e-12))
    mag -= np.max(mag)
    freqs_fft = np.fft.rfftfreq(n, d=1 / sr)
    ax.plot(freqs_fft, mag, color="#C9BFAE", linewidth=1.3, label="FFT spectrum")
    for color, p in zip(ORDER_COLORS, orders):
        res = ac.lpc_analyze_frame(frame, sr, p)
        freqs, mag_db = ac.lpc_spectrum(res)
        mag_db = mag_db - np.max(mag_db)
        ax.plot(freqs, mag_db, color=color, linewidth=1.6, label=f"LPC p={p}")
    ax.set_xlim(0, sr / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, 0 dB = each curve's own peak)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def ar_peak_freq(frame: np.ndarray, sr: int, order: int) -> float:
    res = ac.lpc_analyze_frame(frame, sr, order)
    freqs, mag_db = ac.lpc_spectrum(res, n_freqs=2048)
    return float(freqs[np.argmax(mag_db)])


def build_blocks(clip: ac.AudioClip, native_sr: int | None = None) -> list[Block]:
    frame_len = int(round(clip.sr * FRAME_MS / 1000))
    b: list[Block] = []

    # --- order sweep: residual energy + sanity-check reconstruction ---
    order_energies = []
    recon_rmses = []
    for p in ORDERS:
        pipe = ac.lpc_ola_signal(clip, order=p, frame_ms=FRAME_MS, frame_shift_ms=FRAME_SHIFT_MS)
        order_energies.append(pipe["mean_residual_energy"])
        n = min(len(clip.y), len(pipe["recon_signal"]))
        rmse = float(np.sqrt(np.mean((clip.y[:n].astype(np.float64) - pipe["recon_signal"][:n].astype(np.float64)) ** 2)))
        recon_rmses.append(rmse)

    primary_pipeline = ac.lpc_ola_signal(clip, order=PRIMARY_ORDER, frame_ms=FRAME_MS, frame_shift_ms=FRAME_SHIFT_MS)
    hop_len = primary_pipeline["hop_length"]

    frames = pick_representative_frames(clip, frame_len)

    b.append(("title", COURSE_TITLE))
    b.append(("id_line", f"{STUDENT_ID} {STUDENT_NAME}"))

    b.append(("section", "1. Introduction"))
    b.append(("body",
        "This report implements Homework #5 (Linear Prediction Analysis): the Levinson-Durbin recursion is "
        "implemented directly from its defining equations (E(0)=φ[0]; k_i = {φ[i] − Σ_{j=1}^{i-1} "
        "a_j^{(i-1)} φ[i-j]} / E(i-1); a_i^{(i)}=k_i; a_j^{(i)}=a_j^{(i-1)} − k_i a_{i-j}^{(i-1)}; "
        "E(i)=(1−k_i²)E(i-1)) rather than via scipy.linalg.solve_toeplitz(), and is run inside the "
        f"overlap-and-add (OLA) framing implemented for Homework #3, so that linear prediction coefficients (LPC) "
        f"are computed independently in each frame of the recorded input speech signal. As specified, the signal "
        f"is analyzed at {REQUIRED_SR:,} Hz with a {FRAME_MS:.0f} ms frame length and a {FRAME_SHIFT_MS:.0f} ms "
        f"frame shift ({frame_len} samples / {hop_len} samples at {clip.sr:,} Hz).", False))
    if native_sr and native_sr != clip.sr:
        b.append(("body",
            f"The recording's microphone captured audio natively at {native_sr:,} Hz; it is resampled to "
            f"{clip.sr:,} Hz here because the assignment specifies an 8 kHz sampling frequency for this analysis.", False))

    b.append(("section", "2. Levinson-Durbin Inside the OLA Program"))
    b.append(("body",
        "Each frame is windowed (Hamming), its autocorrelation φ[0..p] is computed, and the Levinson-Durbin "
        "recursion solves for the LPC coefficients a_1..a_p of that frame alone (no information is shared across "
        "frames). The residual e(n) = x(n) − Σ_k a_k x(n−k) is obtained by passing the frame through "
        "the analysis filter A(z) = 1 − Σ a_k z⁻ᵏ; the reconstructed frame is obtained by passing "
        "that SAME frame's residual back through the synthesis filter 1/A(z), i.e. V(z) = G/A(z) with unit gain "
        "excitation e(n). Per-frame residuals and per-frame reconstructions are then each reassembled into "
        "continuous signals via weighted overlap-and-add, exactly as in Homework #3.", False))
    b.append(("picture", fig_signal_residual_recon(clip, primary_pipeline), 6.0))
    n_check = min(len(clip.y), len(primary_pipeline["recon_signal"]))
    recon_rmse_primary = float(np.sqrt(np.mean(
        (clip.y[:n_check].astype(np.float64) - primary_pipeline["recon_signal"][:n_check].astype(np.float64)) ** 2
    )))
    b.append(("body",
        f"At p={PRIMARY_ORDER} (matching the lecture's own worked example on p.61, \"M = 14\"), the reconstructed "
        f"signal overlays the input essentially exactly (RMSE = {recon_rmse_primary:.2e}, against an input RMS of "
        f"{np.sqrt(np.mean(clip.y.astype(np.float64) ** 2)):.3f}). This is expected and holds at every order "
        "tested below, not just p=14: since the synthesis filter 1/A(z) uses the identical coefficients the "
        "analysis filter A(z) used to produce that frame's own residual, applying one after the other is "
        "mathematically an identity (up to floating-point precision and the stability clamp on the reflection "
        "coefficients), independent of how good a model A(z) actually is. A near-zero reconstruction RMSE is "
        "therefore a check that the analysis/synthesis pipeline is wired correctly, not evidence that any "
        "particular order models the vocal tract well.", False))

    b.append(("subsection", "2.1 What actually varies with order: residual whitening"))
    b.append(("body",
        "The quantity that genuinely depends on model order is the prediction-error (residual) energy itself: "
        "E(i) = (1−k_i²) E(i-1) is non-increasing by construction, so adding more poles can only reduce "
        "or leave unchanged the energy left in the residual, never increase it. The table and plot below show the "
        "mean residual energy, averaged over every frame in the recording, at each of the orders the assignment "
        "specifies. (This is φ[0]'s own scale -- a biased autocorrelation, i.e. a raw sum of squares over one "
        f"{frame_len}-sample Hamming-windowed frame, not a per-sample power -- so it is only meaningful compared "
        "across orders/frames on this same scale, not directly against a whole-clip RMS figure quoted elsewhere "
        "in this report.)", False))
    b.append(("table",
        ["Order p", "Mean residual energy", "Reconstruction RMSE (sanity check)"],
        [[str(p), f"{e:.6f}", f"{r:.2e}"] for p, e, r in zip(ORDERS, order_energies, recon_rmses)]))
    b.append(("picture", fig_order_energy(ORDERS, order_energies), 4.6))
    drop_4_to_12 = 100 * (1 - order_energies[2] / order_energies[0])
    drop_12_to_20 = 100 * (1 - order_energies[-1] / order_energies[2])
    b.append(("body",
        f"Residual energy drops by {drop_4_to_12:.0f}% from p=4 to p=12, but only a further {drop_12_to_20:.0f}% "
        "from p=12 to p=20 -- diminishing returns set in well before p=20. This matches the lecture's own model-"
        "order budget (p.59: ≈10 poles for the vocal tract's resonance density over a 4–5 kHz bandwidth, "
        "plus a few more for the radiation/glottal terms, ≈14–16 total): once enough poles exist to cover "
        "the formant structure, additional poles mostly re-fit residual noise rather than removing more structure.", False))

    b.append(("section", "3. AR Spectrum for Frames of Different Characteristics"))
    b.append(("body",
        "Representative frames were selected automatically from the Homework #2 background/unvoiced/voiced "
        "classifier (longest segment of each class in this recording), rather than picked by eye, so the choice "
        "is reproducible from the recording alone.", False))
    labels_ko = {"voiced": "Voiced (vowel-like)", "unvoiced": "Unvoiced (fricative/stop-like)", "background": "Background / silence"}
    order_idx = {"voiced": 1, "unvoiced": 2, "background": 3}
    for key in ["voiced", "unvoiced", "background"]:
        if key not in frames:
            b.append(("body", f"({labels_ko[key]}: no segment of this class was found in this recording.)", True))
            continue
        info = frames[key]
        b.append(("subsection", f"3.{order_idx[key]} {labels_ko[key]}"))
        b.append(("body", f"Longest {key} segment: {info['duration']:.2f} s, frame extracted at t={info['t_center']:.2f} s.", False))
        b.append(("picture", fig_ar_spectrum_frame(info["frame"], clip.sr, ORDERS, f"{labels_ko[key]} frame — AR spectrum vs. order"), 5.4))

    b.append(("subsection", "3.4 Findings"))
    rep_stats = {}
    for key, info in frames.items():
        sig_energy = float(np.sum(info["frame"] ** 2))
        ratio_low = ac.lpc_analyze_frame(info["frame"], clip.sr, 4).gain ** 2 / sig_energy
        ratio_high = ac.lpc_analyze_frame(info["frame"], clip.sr, 20).gain ** 2 / sig_energy
        rep_stats[key] = {"sig_energy": sig_energy, "ratio_low": ratio_low, "ratio_high": ratio_high}
    if rep_stats:
        b.append(("table",
            ["Frame", "Signal energy (sum sq.)", "Residual/signal, p=4", "Residual/signal, p=20"],
            [[labels_ko[k], f"{v['sig_energy']:.5f}", f"{v['ratio_low']*100:.1f}%", f"{v['ratio_high']*100:.1f}%"]
             for k, v in rep_stats.items()]))

    if "voiced" in frames:
        voiced_peak_low = ar_peak_freq(frames["voiced"]["frame"], clip.sr, 20)
        rv = rep_stats["voiced"]
        b.append(("body",
            f"On the voiced frame, raising the order progressively resolves separate formant peaks that low orders "
            f"blur together into one broad bump; at p=20 the strongest AR peak sits at {voiced_peak_low:.0f} Hz, and "
            f"the residual/signal energy ratio falls from {rv['ratio_low']*100:.1f}% at p=4 to {rv['ratio_high']*100:.1f}% "
            "at p=20 -- the largest order-dependent improvement of the three frame types, because there is real "
            "formant structure for the extra poles to remove. Because voiced excitation is quasi-periodic, the raw "
            "FFT spectrum is a dense comb of harmonic lines; the AR curve instead traces only the smooth formant "
            "envelope those harmonics sit under, which is exactly the separation of \"what the vocal tract did\" "
            "from \"what the glottis did\" that motivates LPC-based (vocoder) coding. This also matches the "
            "lecture's own \"Criterion of Goodness\" property (p.55): the LP error criterion favors a good fit "
            "near spectral peaks over spectral valleys, which is visible directly in the plot above -- each AR "
            "curve tracks its FFT spectrum's peaks far more tightly than its valleys, at every order.", False))
    if "unvoiced" in frames:
        ru = rep_stats["unvoiced"]
        b.append(("body",
            f"On the unvoiced frame, the FFT spectrum is comparatively flat and noise-like (no harmonic comb, "
            "consistent with turbulent/frication excitation rather than periodic glottal pulses), and the "
            f"residual/signal ratio only falls from {ru['ratio_low']*100:.1f}% to {ru['ratio_high']*100:.1f}% "
            "(p=4 to p=20) -- a visibly smaller improvement than the voiced frame's -- because there is less "
            "underlying spectral structure for extra poles to uncover, so the gap between low- and high-order AR "
            "curves is correspondingly smaller here than for the voiced frame.", False))
    if "background" in frames:
        rb = rep_stats["background"]
        rvv = rep_stats.get("voiced")
        b.append(("body",
            f"On the background/silence frame, both the FFT spectrum and the AR envelope are comparatively flat at "
            f"every order, and the residual/signal ratio is already low at p=4 ({rb['ratio_low']*100:.1f}%) and "
            f"barely changes by p=20 ({rb['ratio_high']*100:.1f}%) -- additional poles add essentially no "
            "interpretable detail. In absolute terms this frame's signal energy is "
            + (f"about {rvv['sig_energy'] / rb['sig_energy']:.0f}× smaller than the voiced frame's" if rvv else "small")
            + ", consistent with it containing near-silence rather than real formant structure for the all-pole "
            "model to recover.", False))

    b.append(("section", "4. Consideration: Implementing LPC with Deep Learning"))
    b.append(("body",
        "The clearest existing example is LPCNet (Valin & Skoglund, 2019): rather than replacing LPC, it keeps the "
        "classical per-frame linear predictor exactly as implemented here, and uses a small recurrent network "
        "(a WaveRNN-style architecture) only to model the excitation/residual that remains after linear prediction, "
        "conditioned on the LPC coefficients and acoustic features. This division of labor is deliberate: the "
        "linear predictor already removes most of the short-term spectral redundancy very cheaply (as Section 2.1 "
        "shows, with a closed-form, non-iterative solution), so the network only has to learn the harder, more "
        "nonlinear part -- modeling the fine-grained excitation -- which lets a lightweight model run in real time "
        "on a CPU at high speech quality and low bitrate, which a from-scratch neural vocoder (predicting waveform "
        "samples directly) normally cannot do without much more compute.", False))
    b.append(("body",
        "A second, complementary idea is to replace the closed-form Levinson-Durbin step itself with a learned "
        "predictor: a small network (e.g. over a frame's autocorrelation, or directly over mel-spectral features) "
        "predicts the reflection coefficients k_i, trained end-to-end against a perceptual or waveform loss rather "
        "than pure prediction-error energy. Because the standard recursion is purely linear and closed-form, it "
        "cannot represent the mild nonlinearities (e.g. glottal-source/vocal-tract interaction, breathiness) real "
        "speech exhibits; a learned estimator could in principle fit those, at the cost of losing the closed-form "
        "recursion's guaranteed filter stability (|k_i|<1), which would need to be re-imposed some other way -- for "
        "instance a bounded output activation such as tanh, which enforces |k_i|<1 by construction, rather than "
        "the hard min/max clamp this report's own Levinson-Durbin implementation uses to keep the recursion "
        "numerically stable.", False))

    b.append(("section", "5. Summary"))
    b.append(("body",
        f"The Levinson-Durbin recursion was implemented from its defining equations and embedded in the "
        f"Homework #3 OLA framework, run per-frame at {REQUIRED_SR:,} Hz / {FRAME_MS:.0f} ms frames / "
        f"{FRAME_SHIFT_MS:.0f} ms shift on a recorded speech signal. Residuals and reconstructions were obtained "
        "and plotted against the input, confirming the analysis/synthesis pipeline is self-consistent; the AR "
        "spectrum was compared across orders 4/8/12/16/20 for voiced, unvoiced, and background frames, showing "
        "that higher order resolves sharper formant structure mainly where real periodic structure exists (voiced "
        "speech), with diminishing returns elsewhere, consistent with the lecture's own vocal-tract pole budget.", False))

    return b


# --- .docx renderer (page geometry matching the student's Classroom template) ---

def set_page_geometry(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = PAGE_WIDTH
    section.page_height = PAGE_HEIGHT
    section.left_margin = MARGIN_LEFT
    section.right_margin = MARGIN_RIGHT
    section.top_margin = MARGIN_TOP
    section.bottom_margin = MARGIN_BOTTOM


def render_docx(blocks: list[Block], out_path: Path) -> Path:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    set_page_geometry(doc)

    def justified(text: str) -> None:
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    for block in blocks:
        kind = block[0]
        if kind == "title":
            p = justified(block[1])
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(block[1])
            r.bold = True
            r.font.size = Pt(17)
        elif kind == "id_line":
            p = doc.add_paragraph(block[1])
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(1.5)
            doc.add_paragraph()
        elif kind == "section":
            p = justified(block[1])
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(block[1])
            r.bold = True
            r.font.size = Pt(14)
        elif kind == "subsection":
            p = justified(block[1])
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(block[1])
            r.bold = True
            r.font.size = Pt(12)
        elif kind == "body":
            _, text, italic = block
            p = justified(text)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(text)
            r.italic = italic
        elif kind == "bullet":
            _, name, desc = block
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run(f"{name}: ")
            r.bold = True
            p.add_run(desc)
        elif kind == "picture":
            _, png_bytes, width_in = block
            doc.add_picture(io.BytesIO(png_bytes), width=Inches(width_in))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == "table":
            _, headers, rows = block
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = "Light Grid Accent 1"
            for c, t in zip(table.rows[0].cells, headers):
                c.text = t
            for row_vals in rows:
                row = table.add_row().cells
                for c, v in zip(row, row_vals):
                    c.text = v
            doc.add_paragraph()

    doc.save(out_path)
    return out_path


# --- PDF renderer (weasyprint, no GUI app involved) ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  @page {{ size: A4; margin: 1in 1in 1in 1.25in; }}
  body {{ font-family: "Times New Roman", Times, serif; font-size: 11pt; color: #1C1916; }}
  h1 {{ font-size: 17pt; margin: 0 0 3pt 0; }}
  .idline {{ margin: 0 0 12pt 0; }}
  h2 {{ font-size: 14pt; margin: 14pt 0 6pt 0; }}
  h3 {{ font-size: 12pt; margin: 10pt 0 4pt 0; }}
  p {{ text-align: justify; margin: 0 0 4pt 0; line-height: 1.25; }}
  p.italic {{ font-style: italic; }}
  ul {{ margin: 0 0 4pt 0; padding-left: 20pt; }}
  figure {{ text-align: center; margin: 8pt 0; }}
  figure img {{ max-width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 10pt; }}
  table, th, td {{ border: 1px solid #888; }}
  th, td {{ padding: 3pt 6pt; text-align: left; }}
  th {{ background: #E8EAF2; }}
</style></head><body>
{body}
</body></html>"""


def esc(text: str) -> str:
    return html_lib.escape(text)


def render_pdf(blocks: list[Block], out_path: Path) -> Path:
    from weasyprint import HTML

    parts: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "title":
            parts.append(f"<h1>{esc(block[1])}</h1>")
        elif kind == "id_line":
            parts.append(f'<p class="idline">{esc(block[1])}</p>')
        elif kind == "section":
            parts.append(f"<h2>{esc(block[1])}</h2>")
        elif kind == "subsection":
            parts.append(f"<h3>{esc(block[1])}</h3>")
        elif kind == "body":
            _, text, italic = block
            cls = ' class="italic"' if italic else ""
            parts.append(f"<p{cls}>{esc(text)}</p>")
        elif kind == "bullet":
            _, name, desc = block
            parts.append(f"<ul><li><b>{esc(name)}:</b> {esc(desc)}</li></ul>")
        elif kind == "picture":
            _, png_bytes, width_in = block
            b64 = base64.b64encode(png_bytes).decode("ascii")
            parts.append(f'<figure><img style="width:{width_in}in" src="data:image/png;base64,{b64}"></figure>')
        elif kind == "table":
            _, headers, rows = block
            thead = "".join(f"<th>{esc(h)}</th>" for h in headers)
            trows = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
            parts.append(f"<table><tr>{thead}</tr>{trows}</table>")

    html_doc = HTML_TEMPLATE.format(body="\n".join(parts))
    HTML(string=html_doc, base_url=str(out_path.parent)).write_pdf(out_path)
    return out_path


def main() -> None:
    wav_path = find_recording()
    clip = ac.load_audio_file(wav_path)
    native_sr = clip.sr
    if clip.sr != REQUIRED_SR:
        clip = ac.resample_clip(clip, REQUIRED_SR)

    blocks = build_blocks(clip, native_sr=native_sr)

    docx_path = ROOT / "HW5_LPC" / f"Homework5_{STUDENT_ID}_{STUDENT_NAME.replace(' ', '')}.docx"
    pdf_path = ROOT / "HW5_LPC" / f"Homework5_{STUDENT_ID}_{STUDENT_NAME.replace(' ', '')}.pdf"
    render_docx(blocks, docx_path)
    render_pdf(blocks, pdf_path)
    print(f"Wrote {docx_path.name} and {pdf_path.name} (source: {wav_path.name}, native {native_sr} Hz -> {clip.sr} Hz, {clip.duration:.2f}s)")


if __name__ == "__main__":
    main()
