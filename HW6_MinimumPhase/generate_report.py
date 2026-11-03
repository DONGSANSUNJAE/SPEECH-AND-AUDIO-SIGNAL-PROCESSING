"""Build the Homework #6 report: prove that for minimum-phase systems, the
phase is fully determined by the log-magnitude.

Usage:
    .venv/bin/python HW6_MinimumPhase/generate_report.py

Assignment (professor's blackboard, HW #6):
    X(w) = |X(w)| e^(j<X(w)),   log X(w) = log|X(w)| + j<X(w)
    Q: In minimum phase systems, phase terms are fully determined by log
       magnitude terms. Explain/Prove this.

Lab-report format: short numbered steps and tables, no figures, no
narrative prose or self-referential commentary.
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import numpy as np
from docx import Document
from docx.shared import Emu, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent

STUDENT_ID = "2026311323"
STUDENT_NAME = "Howon Chung"

PAGE_WIDTH = Emu(7560310)
PAGE_HEIGHT = Emu(10692130)
MARGIN_LEFT = Emu(1143000)
MARGIN_RIGHT = Emu(914400)
MARGIN_TOP = Emu(914400)
MARGIN_BOTTOM = Emu(914400)
COURSE_TITLE = "Audio and Signal Processing"


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

Block = tuple


def minimum_phase_example(N: int = 4096) -> dict:
    a1, a2 = 0.6, -0.4
    x_min = np.convolve([1, -a1], [1, -a2])
    x_nonmin = np.convolve([-a1, 1], [1, -a2])

    X_min = np.fft.fft(x_min, N)
    X_nonmin = np.fft.fft(x_nonmin, N)
    phase_min = np.unwrap(np.angle(X_min))
    phase_nonmin = np.unwrap(np.angle(X_nonmin))
    logmag_min = np.log(np.maximum(np.abs(X_min), 1e-300))
    logmag_nonmin = np.log(np.maximum(np.abs(X_nonmin), 1e-300))

    cep_min = np.real(np.fft.ifft(logmag_min + 1j * phase_min))
    cep_nonmin = np.real(np.fft.ifft(logmag_nonmin + 1j * phase_nonmin))

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
        "a1": a1, "a2": a2,
        "max_cep_min_neg": float(np.max(np.abs(cep_min[N // 2 + 1:]))),
        "max_cep_min_pos": float(np.max(np.abs(cep_min[:half]))),
        "max_cep_nonmin_neg": float(np.max(np.abs(cep_nonmin[N // 2 + 1:]))),
        "max_cep_nonmin_pos": float(np.max(np.abs(cep_nonmin[:half]))),
        "logmag_diff": float(np.max(np.abs(logmag_min[:half] - logmag_nonmin[:half]))),
        "phase_diff": float(np.max(np.abs(phase_min[:half] - phase_nonmin[:half]))),
        "recon_phase_err": float(np.max(np.abs(phase_recon[:half] - phase_min[:half]))),
        "recon_logmag_err": float(np.max(np.abs(np.log(np.abs(X_recon[:half])) - logmag_min[:half]))),
    }


def build_blocks() -> list[Block]:
    b: list[Block] = []
    ex = minimum_phase_example()

    b.append(("title", COURSE_TITLE))
    b.append(("id_line", f"{STUDENT_ID} {STUDENT_NAME}"))

    b.append(("section", "1. Problem"))
    b.append(("bullet", "Given", "X(ω) = |X(ω)| e^(j∠X(ω)),  log X(ω) = log|X(ω)| + j∠X(ω)"))
    b.append(("bullet", "To prove", "for a minimum-phase system, ∠X(ω) is fully determined by log|X(ω)|"))
    b.append(("bullet", "Minimum phase", "every zero and every pole of X(z) is strictly inside the unit circle"))

    b.append(("section", "2. Setup"))
    b.append(("bullet", "Complex cepstrum", "x̂[n] = IDFT{log X(k)}"))
    b.append(("bullet", "Step 1", "minimum phase ⟹ x̂[n] is causal (x̂[n]=0 for n<0)"))
    b.append(("bullet", "Step 2", "any real, causal h[n] has Re{H(ω)} and Im{H(ω)} related by the discrete Hilbert transform"))
    b.append(("bullet", "Combine", "apply Step 2 to h[n]=x̂[n] using Step 1's causality"))

    b.append(("section", "3. Step 1 — Minimum Phase ⟹ Causal Cepstrum"))
    b.append(("eq", "X(z) = A · Π_k (1 − a_k z⁻¹) / Π_k (1 − c_k z⁻¹),   |a_k| < 1, |c_k| < 1 for all k"))
    b.append(("eq", "log X(z) = log A + Σ_k log(1 − a_k z⁻¹) − Σ_k log(1 − c_k z⁻¹)"))
    b.append(("eq", "log(1 − a z⁻¹) = −Σ_{ℓ=1}^∞ (aℓ/ℓ) z⁻ℓ     (valid for |z| > |a|, true since |a| < 1)"))
    b.append(("bullet", "Only negative powers", "every term above is z⁻ℓ (ℓ≥1) — no z⁺ℓ term appears"))
    b.append(("eq", "log X(z) = log A + Σ_{n=1}^∞ x̂[n] z⁻ⁿ,   x̂[n] = (Σ_k a_kⁿ − Σ_k c_kⁿ)/n for n ≥ 1"))
    b.append(("bullet", "Result", "x̂[n] = 0 for n < 0 — causal. (A zero/pole outside the unit circle gives the same "
                                   "expansion only positive powers of z instead — an anticausal contribution.)"))

    b.append(("section", "4. Step 2 — Causal Real Sequence ⟹ Hilbert Pair"))
    b.append(("bullet", "Setup", "h[n] real, h[n]=0 for n<0. Even/odd split: h_e[n]=(h[n]+h[−n])/2, h_o[n]=(h[n]−h[−n])/2"))
    b.append(("eq", "n > 0:  h[−n]=0  ⟹  h_e[n] = h_o[n] = h[n]/2"))
    b.append(("eq", "n < 0:  h[n]=0  ⟹  h_e[n] = h[−n]/2 = −h_o[n]"))
    b.append(("bullet", "Combined", "h_o[n] = sgn(n) · h_e[n] for all n ≠ 0"))
    b.append(("bullet", "DTFT", "h_e even+real ⟹ H_e(ω) real;  h_o odd+real ⟹ H_o(ω) purely imaginary"))
    b.append(("bullet", "Identify", "Re{H(ω)} = H_e(ω),  Im{H(ω)} comes from H_o(ω)"))
    b.append(("bullet", "Result", "multiplying by sgn(n) in time = Hilbert transform in frequency, so "
                                   "Im{H(ω)} = Hilbert{ Re{H(ω)} } for any real causal h[n]"))

    b.append(("section", "5. Combine"))
    b.append(("eq", "Re{log X(ω)} = log|X(ω)|,   Im{log X(ω)} = ∠X(ω)"))
    b.append(("eq", "⟹   ∠X(ω) = Hilbert{ log|X(ω)| }"))
    b.append(("bullet", "Conclusion", "phase is the Hilbert transform of log-magnitude for minimum-phase systems — a "
                                       "fixed, invertible linear map, except for x̂[0]=log A (overall gain, no phase info)"))

    b.append(("section", "6. Why Minimum Phase Is Required"))
    b.append(("bullet", "Zero/pole outside unit circle", "contributes only to n<0 (anticausal) part of x̂[n]"))
    b.append(("bullet", "General (mixed-phase) system", "cepstrum has both causal and anticausal parts — Step 2 no longer applies"))
    b.append(("bullet", "Consequence", "multiple phase responses can share the same |X(ω)| (all-pass ambiguity); "
                                        "minimum phase removes this"))
    b.append(("bullet", "Reconstruction (Ch.6 p.31 liftering)", "from log|X(ω)| alone: IDFT → real cepstrum (even); "
                                        "keep n=0, double n>0, zero n<0 → causal x̂[n]; DFT + exp → full X(ω)"))

    b.append(("section", "7. Numerical Verification"))
    b.append(("bullet", "Example", f"X(z) = (1−a₁z⁻¹)(1−a₂z⁻¹), a₁={ex['a1']}, a₂={ex['a2']} (minimum phase)"))
    b.append(("bullet", "Contrast", f"a₁'s zero reflected to 1/a₁ (outside unit circle); same |X(ω)| "
                                     f"(max diff = {ex['logmag_diff']:.1e})"))
    b.append(("table",
        ["Quantity", "Minimum phase", "Non-minimum phase (same |X(ω)|)"],
        [
            ["max |x̂[n]|, n<0", f"{ex['max_cep_min_neg']:.1e}", f"{ex['max_cep_nonmin_neg']:.3f}"],
            ["max |x̂[n]|, n≥0", f"{ex['max_cep_min_pos']:.3f}", f"{ex['max_cep_nonmin_pos']:.3f}"],
        ]))
    b.append(("bullet", "Causality check", "min-phase n<0 term ≈ 0 (machine precision); non-min-phase n<0 term is "
                                            "the same order as its n≥0 term — not causal"))
    b.append(("bullet", "Phase difference", f"{ex['phase_diff']:.3f} rad max, despite identical magnitude"))
    b.append(("table",
        ["Reconstruction check (from log|X(ω)| alone)", "Error vs. true value"],
        [
            ["Phase", f"{ex['recon_phase_err']:.1e} rad"],
            ["Log-magnitude", f"{ex['recon_logmag_err']:.1e}"],
        ]))

    b.append(("section", "8. Conclusion"))
    b.append(("bullet", "Minimum phase", "all zeros/poles inside unit circle ⟹ cepstrum causal"))
    b.append(("bullet", "Causal + real", "⟹ Re/Im of transform are a Hilbert pair (general fact, any causal real sequence)"))
    b.append(("bullet", "Applied to log X(ω)", "∠X(ω) = Hilbert{log|X(ω)|} — phase fully determined by log-magnitude"))
    b.append(("bullet", "Breaks without minimum phase", "zeros/poles outside the unit circle add an anticausal "
                                                          "cepstrum part, verified numerically above"))

    return b


# --- .docx renderer ---

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
        elif kind == "eq":
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(block[1])
            r.italic = True
        elif kind == "bullet":
            _, name, desc = block
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run(f"{name}: ")
            r.bold = True
            p.add_run(desc)
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


# --- PDF renderer (weasyprint) ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  @page {{ size: A4; margin: 1in 1in 1in 1.25in; }}
  body {{ font-family: "Times New Roman", Times, serif; font-size: 11pt; color: #1C1916; }}
  h1 {{ font-size: 17pt; margin: 0 0 3pt 0; }}
  .idline {{ margin: 0 0 12pt 0; }}
  h2 {{ font-size: 14pt; margin: 14pt 0 6pt 0; }}
  p.eq {{ text-align: center; font-style: italic; margin: 4pt 0; }}
  ul {{ margin: 0 0 4pt 0; padding-left: 20pt; }}
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
        elif kind == "eq":
            parts.append(f'<p class="eq">{esc(block[1])}</p>')
        elif kind == "bullet":
            _, name, desc = block
            parts.append(f"<ul><li><b>{esc(name)}:</b> {esc(desc)}</li></ul>")
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
