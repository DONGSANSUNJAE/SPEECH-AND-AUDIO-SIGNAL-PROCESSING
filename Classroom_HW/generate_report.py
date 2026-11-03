"""Build the Homework #6 report: prove that for minimum-phase systems, the
phase is fully determined by the log-magnitude.

Usage:
    .venv/bin/python Classroom_HW/generate_report.py

Assignment (from the professor's blackboard, HW #6):
    X(w) = |X(w)| e^(j<X(w)),   log X(w) = log|X(w)| + j<X(w)
    Q: In minimum phase systems, phase terms are fully determined by log
       magnitude terms. Explain/Prove this.

The proof is built entirely from Chapter6.pdf's own material:
  - p.7  (Discrete Complex Cepstrum): the log X(k) = log|X(k)| + j<X(k)
    definition and the complex cepstrum x-hat[n] = IDFT{log X(k)}.
  - p.12 (Rational z-transforms): a zero/pole strictly inside the unit
    circle contributes only non-negative powers of z^-1 to log X(z)'s
    power series -- the key lemma generalized here to the full
    minimum-phase case (ALL zeros/poles inside the unit circle).
  - p.31 (Zero- and Minimum-phase Synthesis): the liftering reconstruction
    this proof's result makes possible, used directly as the numerical
    verification method below.

All content (text/figures/tables) is computed once into format-agnostic
"blocks", then rendered to both .docx and .pdf (same harness as
HW5_LPC/generate_report.py). weasyprint needs Homebrew's pango/cairo and
DYLD_FALLBACK_LIBRARY_PATH, as documented there.
"""

from __future__ import annotations

import base64
import html as html_lib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Emu, Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent

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


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Content blocks (same format-agnostic scheme as HW5_LPC/generate_report.py)
# ---------------------------------------------------------------------------

Block = tuple


# --- numerical example: a minimum-phase 2nd-order FIR, and a non-minimum- ---
# --- phase system with the IDENTICAL magnitude response for contrast.    ---

def minimum_phase_example(N: int = 4096):
    a1, a2 = 0.6, -0.4  # both zeros strictly inside the unit circle
    x_min = np.convolve([1, -a1], [1, -a2])
    x_nonmin = np.convolve([-a1, 1], [1, -a2])  # a1's zero reflected OUTSIDE (1/a1)

    X_min = np.fft.fft(x_min, N)
    X_nonmin = np.fft.fft(x_nonmin, N)
    phase_min = np.unwrap(np.angle(X_min))
    phase_nonmin = np.unwrap(np.angle(X_nonmin))
    logmag_min = np.log(np.maximum(np.abs(X_min), 1e-300))
    logmag_nonmin = np.log(np.maximum(np.abs(X_nonmin), 1e-300))

    cep_min = np.real(np.fft.ifft(logmag_min + 1j * phase_min))
    cep_nonmin = np.real(np.fft.ifft(logmag_nonmin + 1j * phase_nonmin))

    # reconstruct phase from the MINIMUM-PHASE case's log-magnitude ALONE,
    # via the liftering procedure (Chapter6 p.31): causal part of the real
    # cepstrum, doubled, zero for n<0.
    real_ceps = np.real(np.fft.ifft(logmag_min))
    lifter = np.zeros(N)
    lifter[0] = 1.0
    lifter[1:N // 2] = 2.0
    lifter[N // 2] = 1.0
    xhat_recon = real_ceps * lifter
    X_recon = np.exp(np.fft.fft(xhat_recon))
    phase_recon = np.unwrap(np.angle(X_recon))

    half = N // 2 + 1
    return {
        "N": N, "a1": a1, "a2": a2,
        "x_min": x_min, "x_nonmin": x_nonmin,
        "cep_min": cep_min, "cep_nonmin": cep_nonmin,
        "max_cep_min_neg": float(np.max(np.abs(cep_min[N // 2 + 1:]))),
        "max_cep_min_pos": float(np.max(np.abs(cep_min[:half]))),
        "max_cep_nonmin_neg": float(np.max(np.abs(cep_nonmin[N // 2 + 1:]))),
        "max_cep_nonmin_pos": float(np.max(np.abs(cep_nonmin[:half]))),
        "logmag_diff": float(np.max(np.abs(logmag_min[:half] - logmag_nonmin[:half]))),
        "phase_diff_min_nonmin": float(np.max(np.abs(phase_min[:half] - phase_nonmin[:half]))),
        "recon_phase_err": float(np.max(np.abs(phase_recon[:half] - phase_min[:half]))),
        "recon_logmag_err": float(np.max(np.abs(np.log(np.abs(X_recon[:half])) - logmag_min[:half]))),
        "phase_min": phase_min[:half], "phase_nonmin": phase_nonmin[:half], "phase_recon": phase_recon[:half],
    }


def fig_cepstrum_causality(ex: dict) -> bytes:
    N = ex["N"]
    n = np.arange(-20, 21)
    idx = n % N
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    axes[0].stem(n, ex["cep_min"][idx])
    axes[0].axvline(0, color="#A8481C", linewidth=0.8, linestyle="--")
    axes[0].set_title("Minimum-phase: cepstrum is causal", fontsize=10, loc="left")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("x̂[n]")
    axes[1].stem(n, ex["cep_nonmin"][idx])
    axes[1].axvline(0, color="#A8481C", linewidth=0.8, linestyle="--")
    axes[1].set_title("Same |X(ω)|, non-min-phase: NOT causal", fontsize=10, loc="left")
    axes[1].set_xlabel("n")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def fig_phase_reconstruction(ex: dict) -> bytes:
    half = len(ex["phase_min"])
    w = np.linspace(0, np.pi, half)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(w, ex["phase_min"], color="#C9BFAE", linewidth=3.0, label="true phase (minimum-phase system)")
    ax.plot(w, ex["phase_recon"], color="#1B3A6B", linewidth=1.2, linestyle="--",
            label="reconstructed from log|X(ω)| alone")
    ax.plot(w, ex["phase_nonmin"], color="#A8481C", linewidth=1.2,
            label="non-min-phase system (same |X(ω)|, different phase)")
    ax.set_xlabel("ω (rad)")
    ax.set_ylabel("phase (rad)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Phase: true vs. reconstructed-from-magnitude vs. a same-magnitude alternative", fontsize=10, loc="left")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def build_blocks() -> list[Block]:
    b: list[Block] = []
    ex = minimum_phase_example()

    b.append(("title", COURSE_TITLE))
    b.append(("id_line", f"{STUDENT_ID} {STUDENT_NAME}"))

    b.append(("section", "1. Problem Statement"))
    b.append(("body",
        "Given X(ω) = |X(ω)| e^(j∠X(ω)), so that log X(ω) = log|X(ω)| + j∠X(ω): prove that for a minimum-phase "
        "system, the phase ∠X(ω) is fully determined by the log-magnitude log|X(ω)|.", False))
    b.append(("body",
        "A minimum-phase sequence x[n] is real, causal, and stable, with a causal and stable inverse 1/X(z) -- "
        "equivalently, every zero AND every pole of X(z) lies strictly inside the unit circle.", False))

    b.append(("section", "2. Setup: the Complex Cepstrum"))
    b.append(("body",
        "Define the complex cepstrum as the inverse transform of log X: x̂[n] = IDFT{log X(k)}, where "
        "log X(k) = log|X(k)| + j∠X(k). The proof proceeds in two steps: (1) show that the minimum-phase "
        "condition forces x̂[n] to be causal (x̂[n] = 0 for n < 0); (2) show that for ANY real, causal sequence, "
        "the real and imaginary parts of its transform are a Hilbert-transform pair. Combining the two: "
        "log|X(ω)| and ∠X(ω) are a Hilbert-transform pair whenever x[n] is minimum phase, which is exactly the "
        "sense in which one determines the other.", False))

    b.append(("section", "3. Step 1 — Minimum Phase ⟹ Causal Cepstrum"))
    b.append(("body",
        "Write X(z) in factored (zero/pole) form. For a minimum-phase system, every zero a_k and every pole c_k "
        "satisfies |a_k| < 1, |c_k| < 1 (strictly inside the unit circle):", False))
    b.append(("body", "X(z) = A · Π_k (1 − a_k z⁻¹) / Π_k (1 − c_k z⁻¹),   |a_k| < 1,  |c_k| < 1 for all k", True))
    b.append(("body",
        "Taking the log turns the product into a sum: log X(z) = log A + Σ_k log(1 − a_k z⁻¹) − Σ_k log(1 − c_k z⁻¹). "
        "Each term expands as a power series using log(1−x) = −Σ_{ℓ=1}^∞ xℓ/ℓ, valid for |x|<1, i.e. for |z| > |a_k| "
        "— a region that includes the unit circle since |a_k| < 1:", False))
    b.append(("body", "log(1 − a_k z⁻¹) = −Σ_{ℓ=1}^∞ (a_kℓ/ℓ) z⁻ℓ", True))
    b.append(("body",
        "Every term on the right is a NEGATIVE power of z (z⁻ℓ, ℓ ≥ 1) — there is no z⁺ℓ term at all. (Had a_k "
        "instead been OUTSIDE the unit circle, the same expansion, now valid for |z| < |a_k|, would have produced "
        "only POSITIVE powers of z, i.e. an anticausal contribution — this is the exact mechanism that breaks "
        "down outside the minimum-phase case, used again in Section 6.) Summing every zero and pole term:", False))
    b.append(("body", "log X(z) = log A + Σ_{n=1}^∞ x̂[n] z⁻ⁿ,   x̂[n] = (Σ_k a_kⁿ − Σ_k c_kⁿ) / n  for n ≥ 1", True))
    b.append(("body",
        "No negative-n term appears anywhere in this sum. Therefore x̂[n] = 0 for n < 0: the complex cepstrum of "
        "a minimum-phase sequence is causal. This is the same expansion Chapter 6 (p.12) derives for a single "
        "zero factor, generalized here to the full minimum-phase product (every factor, not just one).", False))

    b.append(("section", "4. Step 2 — A Causal Real Sequence's Transform Has Hilbert-Paired Real/Imaginary Parts"))
    b.append(("body",
        "This step uses no properties of X specifically — it holds for any real, causal sequence h[n] (h[n]=0 "
        "for n<0). Split h[n] into even and odd parts: h_e[n] = (h[n]+h[−n])/2, h_o[n] = (h[n]−h[−n])/2. Because "
        "h is causal:", False))
    b.append(("body", "n > 0:  h[−n] = 0  ⟹  h_e[n] = h_o[n] = h[n]/2\nn < 0:  h[n] = 0  ⟹  h_e[n] = h[−n]/2 = −h_o[n]",
               True))
    b.append(("body",
        "Both cases combine into a single relation valid for every n ≠ 0: h_o[n] = sgn(n) · h_e[n]. Taking the "
        "DTFT: h_e[n] is even and real, so its transform H_e(ω) is real (and even); h_o[n] is odd and real, so "
        "its transform H_o(ω) is purely imaginary (and odd). Hence Re{H(ω)} comes entirely from h_e, and "
        "Im{H(ω)} entirely from h_o. The time-domain relation h_o[n] = sgn(n)·h_e[n] is precisely the operation "
        "that defines the discrete Hilbert transform in the frequency domain: multiplying a sequence by sgn(n) "
        "is the time-domain counterpart of taking its Hilbert transform. So for ANY real causal h[n]: "
        "Im{H(ω)} = Hilbert{Re{H(ω)}} (up to the fixed sign convention of the transform).", False))

    b.append(("section", "5. Combining the Two Steps"))
    b.append(("body",
        "Apply Step 2 with h[n] = x̂[n]. By Step 1, x̂[n] is causal whenever x[n] is minimum phase, so Step 2's "
        "conclusion applies directly:", False))
    b.append(("body", "Re{log X(ω)} = log|X(ω)|,   Im{log X(ω)} = ∠X(ω)\n⟹   ∠X(ω) = Hilbert{ log|X(ω)| }",
               True))
    b.append(("body",
        "This is the statement to prove: for a minimum-phase system, the phase is the (discrete) Hilbert "
        "transform of the log-magnitude — a fixed, invertible linear operation, not merely a loose correlation. "
        "Knowing log|X(ω)| at every ω therefore determines ∠X(ω) completely (up to the n=0 cepstrum term x̂[0] "
        "= log A, an overall real gain that carries no phase information).", False))

    b.append(("section", "6. Why Minimum Phase Specifically — and the Reconstruction It Enables"))
    b.append(("body",
        "Step 1 used BOTH directions of the zero/pole argument: a factor inside the unit circle contributes only "
        "to n ≥ 0, and (by the same expansion, now centered outside the unit circle) a factor outside contributes "
        "only to n < 0. A general (mixed-phase) system has some zeros/poles outside, so its cepstrum has a "
        "genuine non-causal part, and Step 2 no longer applies — the log-magnitude and phase are then "
        "independent: many different phase responses are consistent with the same |X(ω)| (the differences are "
        "exactly the all-pass factors obtained by reflecting a zero from inside to outside the unit circle, "
        "which leave |X(ω)| unchanged but change ∠X(ω)). Minimum phase is what removes this ambiguity.", False))
    b.append(("body",
        "The proof is also constructive, matching the liftering procedure in Chapter 6 (p.31, \"Zero- and "
        "Minimum-phase Synthesis\"): from log|X(ω)| alone, (i) inverse-transform to get the real cepstrum "
        "(automatically even, since log|X(ω)| is real and even); (ii) keep the n=0 term, double every n>0 term, "
        "discard every n<0 term — this constructs a fully causal x̂[n] satisfying x̂_o[n]=sgn(n)x̂_e[n] exactly; "
        "(iii) forward-transform and exponentiate. The result's phase is not merely similar to the original -- "
        "it recovers the exact minimum-phase ∠X(ω), as verified numerically below.", False))

    b.append(("section", "7. Numerical Verification"))
    b.append(("body",
        f"Example: X(z) = (1 − a₁z⁻¹)(1 − a₂z⁻¹) with a₁={ex['a1']}, a₂={ex['a2']} (both zeros strictly inside "
        f"the unit circle — minimum phase). For contrast, a second system reflects a₁'s zero to 1/a₁ (outside "
        f"the unit circle) while keeping a₂ unchanged; by construction this has the IDENTICAL magnitude response "
        f"(max |log-magnitude difference| = {ex['logmag_diff']:.1e}) but is NOT minimum phase.", False))
    b.append(("table",
        ["Quantity", "Minimum-phase system", "Non-minimum-phase system (same |X(ω)|)"],
        [
            ["max |x̂[n]| for n < 0 (should be 0 iff causal)", f"{ex['max_cep_min_neg']:.2e}", f"{ex['max_cep_nonmin_neg']:.3f}"],
            ["max |x̂[n]| for n ≥ 0", f"{ex['max_cep_min_pos']:.3f}", f"{ex['max_cep_nonmin_pos']:.3f}"],
        ]))
    b.append(("picture", fig_cepstrum_causality(ex), 6.0))
    b.append(("body",
        f"The minimum-phase cepstrum's n<0 part is at machine-precision noise level ({ex['max_cep_min_neg']:.1e}) "
        f"against an n≥0 part of order {ex['max_cep_min_pos']:.1f} -- causal, as Section 3 predicts. The "
        f"non-minimum-phase cepstrum's n<0 part ({ex['max_cep_nonmin_neg']:.3f}) is the same ORDER OF MAGNITUDE "
        "as its n≥0 part -- genuinely non-causal, confirming the minimum-phase condition is what makes "
        "causality (and therefore the Hilbert-transform / determinacy result) hold.", False))
    b.append(("body",
        f"Despite having identical magnitude, the two systems' phases differ by up to {ex['phase_diff_min_nonmin']:.3f} "
        "rad (= π, exactly a half-cycle at ω=π in this example) -- concrete confirmation that log-magnitude alone "
        "does NOT determine phase once the minimum-phase restriction is dropped.", False))
    b.append(("picture", fig_phase_reconstruction(ex), 6.0))
    b.append(("body",
        f"Reconstructing phase from the minimum-phase system's log-magnitude alone (Section 6's liftering "
        f"procedure) matches the true phase to within {ex['recon_phase_err']:.1e} rad, and the reconstructed "
        f"log-magnitude matches to within {ex['recon_logmag_err']:.1e} -- both at floating-point noise level, "
        "i.e. an exact reconstruction up to numerical precision.", False))

    b.append(("section", "8. Conclusion"))
    b.append(("body",
        "For a minimum-phase system, every zero and pole lies inside the unit circle, which forces the complex "
        "cepstrum to be causal (Section 3). Any real, causal sequence has real and imaginary transform parts "
        "related by the discrete Hilbert transform (Section 4) -- a general fact independent of this problem. "
        "Applying this to log X(ω) = log|X(ω)| + j∠X(ω) gives ∠X(ω) = Hilbert{log|X(ω)|}: the phase is the "
        "Hilbert transform of the log-magnitude, hence fully (and constructively) determined by it. The result "
        "fails outside the minimum-phase case because zeros/poles outside the unit circle contribute an "
        "anticausal part to the cepstrum, breaking the causality Step 1 depends on -- confirmed numerically "
        "above by a same-magnitude, different-phase counterexample.", False))

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

    def justified(text: str):
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
            for i, line in enumerate(text.split("\n")):
                if i > 0:
                    p.add_run().add_break()
                r = p.add_run(line)
                r.italic = italic
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
  p {{ text-align: justify; margin: 0 0 4pt 0; line-height: 1.25; white-space: pre-line; }}
  p.italic {{ font-style: italic; }}
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
    blocks = build_blocks()
    docx_path = ROOT / f"Homework6_{STUDENT_ID}_{STUDENT_NAME.replace(' ', '')}.docx"
    pdf_path = ROOT / f"Homework6_{STUDENT_ID}_{STUDENT_NAME.replace(' ', '')}.pdf"
    render_docx(blocks, docx_path)
    render_pdf(blocks, pdf_path)
    print(f"Wrote {docx_path.name} and {pdf_path.name}")


if __name__ == "__main__":
    main()
