#!/usr/bin/env python3
"""
preprocess_pdf.py — HealthPlus PDF Table Fixer

Converts the raw HealthPlus_Company_Guide.pdf into a clean .txt file where:
  • Tables are reconstructed into readable key-value text lines
  • Section headers are preserved
  • Pricing table → "Cardiology: 350 SAR"
  • Diagnostic table → "MRI (1.5T & 3T): Same day — Main centers"

Usage:
    python src/scripts/preprocess_pdf.py \
        --input  assets/HealthPlus_Company_Guide.pdf \
        --output assets/HealthPlus_Company_Guide_clean.txt

Then re-upload the .txt file to the RAG system for better chunking.
"""
import argparse
import re
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Known table schemas ────────────────────────────────────────────────────────
# Matches the known pricing rows in the HealthPlus document.
# Pattern: a specialty name on one line, a number on the next.
PRICING_SPECIALTIES = {
    "General Medicine", "General Practice",
    "Internal Medicine", "Cardiology", "Orthopedics",
    "Oncology", "Gynecology & Obstetrics", "Gynecology",
    "Obstetrics", "Pediatrics", "Dermatology", "Ophthalmology",
}

# Matches the known diagnostic rows (test name, turnaround, location).
DIAGNOSTIC_TESTS = {
    "MRI", "CT Scan", "Digital X-Ray", "X-Ray", "Ultrasound",
    "Genetic Testing", "PCR", "PCR / Molecular Tests",
    "CBC", "Lipid Panel", "HbA1c", "Vitamin D",
}


def _is_integer(text: str) -> bool:
    return bool(re.fullmatch(r"\d+", text.strip()))


def _is_pricing_specialty(text: str) -> bool:
    return text.strip() in PRICING_SPECIALTIES


def _is_diagnostic_test(text: str) -> bool:
    t = text.strip()
    return any(t.startswith(d) for d in DIAGNOSTIC_TESTS)


def reconstruct_tables(raw_lines: list[str]) -> list[str]:
    """
    Walk through raw extracted lines and rebuild broken tables into readable text.

    Heuristic 1 — Pricing table:
        If a line is a known specialty name and the next line is a plain integer,
        emit: "The consultation fee for <Specialty> is <N> SAR for self-pay patients."

    Heuristic 2 — Diagnostic table:
        If a line starts with a known diagnostic test name, collect the next
        2 lines as turnaround and location and emit:
        "The turnaround time for an <Test> scan is <turnaround> and available at <location>."
    """
    out = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()

        # ── Pricing table reconstruction ──────────────────────────────────────
        if _is_pricing_specialty(line) and i + 1 < len(raw_lines):
            next_line = raw_lines[i + 1].strip()
            if _is_integer(next_line):
                out.append(f"The consultation fee for {line} is {next_line} SAR for self-pay patients.")
                i += 2
                continue

        # ── Diagnostic table reconstruction ───────────────────────────────────
        if _is_diagnostic_test(line) and i + 2 < len(raw_lines):
            turnaround = raw_lines[i + 1].strip()
            location = raw_lines[i + 2].strip()
            # Only merge if the next lines look like metadata (not headers)
            if turnaround and location and not turnaround.isupper():
                out.append(f"The turnaround time for {line} is {turnaround.lower()} and available at {location.lower()}.")
                i += 3
                continue

        out.append(line)
        i += 1

    return out


def add_section_markers(lines: list[str]) -> list[str]:
    """
    Insert blank lines before detected section headers so the chunker
    respects natural section boundaries.

    A line is treated as a section header if:
      - It is in ALL CAPS or Title Case with no trailing punctuation
      - It is relatively short (< 80 chars)
      - The previous line is blank
    """
    out = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Heuristic: short title-case or upper-case lines without periods
        if (
            stripped
            and len(stripped) < 80
            and not stripped.endswith(".")
            and not stripped.endswith(",")
            and (stripped.istitle() or stripped.isupper())
        ):
            # Add a double blank line before the header for clean chunking
            if out and out[-1] != "":
                out.append("")
            out.append(stripped)
            out.append("")
        else:
            out.append(line)

    return out


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append(f"\n\n{'='*60}\nPAGE {page_num}\n{'='*60}\n\n{text}")
    doc.close()
    return "\n".join(pages)


def process_pdf(input_path: Path, output_path: Path) -> None:
    logger.info("Extracting text from: %s", input_path)
    raw_text = extract_text_from_pdf(input_path)

    raw_lines = raw_text.splitlines()
    logger.info("Raw lines extracted: %d", len(raw_lines))

    logger.info("Reconstructing broken tables...")
    processed_lines = reconstruct_tables(raw_lines)

    logger.info("Adding section markers...")
    processed_lines = add_section_markers(processed_lines)

    logger.info("Stripping page separator markers...")
    # The ===PAGE N=== lines injected by extract_text_from_pdf are useful for
    # human review but pollute chunk content in the vector DB. Strip them here.
    import re as _re
    page_sep_pattern = _re.compile(r"^={10,}$")
    page_num_pattern = _re.compile(r"^PAGE\s+\d+$")
    processed_lines = [
        l for l in processed_lines
        if not page_sep_pattern.match(l.strip()) and not page_num_pattern.match(l.strip())
    ]

    # Collapse 3+ consecutive blank lines to 2 blank lines max
    final_lines = []
    blank_count = 0
    for line in processed_lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                final_lines.append(line)
        else:
            blank_count = 0
            final_lines.append(line)

    output_text = "\n".join(final_lines)
    output_path.write_text(output_text, encoding="utf-8")
    logger.info("Processed text written to: %s (%d lines)", output_path, len(final_lines))

    # Print a preview of reconstructed table lines for verification
    table_lines = [l for l in final_lines if " SAR" in l or " — " in l]
    if table_lines:
        logger.info("=== Table Reconstructions (preview) ===")
        for tl in table_lines[:20]:
            logger.info("  %s", tl)


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDF for RAG ingestion.")
    parser.add_argument("--input", required=True, help="Path to source PDF file")
    parser.add_argument("--output", required=True, help="Path for cleaned .txt output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_pdf(input_path, output_path)
    logger.info("Done. Upload the .txt file to the RAG system and reprocess.")


if __name__ == "__main__":
    main()
