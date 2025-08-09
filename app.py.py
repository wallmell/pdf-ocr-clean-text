# Streamlit PDF → Clean Text (with OCR) Web App
# -------------------------------------------------
# This app extracts ONLY the main text from PDFs (native or scanned) and
# excludes images, charts, page numbers, repeating headers/footers, and footnotes.
# It is designed to run on Streamlit Community Cloud (no system Tesseract needed).
# OCR is done with EasyOCR (PyTorch), and native text is read via PyMuPDF.

import io
import re
from typing import List, Tuple, Dict

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

_EASYOCR_READER = None

def get_easyocr_reader(lang_codes: List[str]):
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(lang_codes, gpu=False)
    return _EASYOCR_READER

def normalize_line(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_probable_page_number(line: str) -> bool:
    patterns = [
        re.compile(r"^\s*\d+\s*$"),
        re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
        re.compile(r"^\s*page\s*\d+\s*$", re.I),
        re.compile(r"^\s*\d+\s*of\s*\d+\s*$", re.I)
    ]
    return any(pat.match(line) for pat in patterns)

def gather_repeating_headers_footers(pages_lines: List[List[str]], sample_lines=3, threshold=0.5):
    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}
    total_pages = len(pages_lines)
    for lines in pages_lines:
        tops = lines[:sample_lines]
        bots = lines[-sample_lines:] if len(lines) >= sample_lines else lines[-1:]
        for t in tops:
            top_counts[t] = top_counts.get(t, 0) + 1
        for b in bots:
            bot_counts[b] = bot_counts.get(b, 0) + 1
    top_repeats = {t for t, c in top_counts.items() if c / total_pages >= threshold}
    bot_repeats = {b for b, c in bot_counts.items() if c / total_pages >= threshold}
    return top_repeats, bot_repeats

def stitch_paragraphs(lines: List[str]) -> str:
    out = []
    buf = ""
    for line in lines:
        if not line:
            if buf:
                out.append(buf)
                buf = ""
            continue
        if buf.endswith("-"):
            buf = buf[:-1] + line
        else:
            if buf:
                if re.match(r"^\s{2,}\S", line) or len(line) <= 3:
                    out.append(buf)
                    buf = line
                else:
                    buf += " " + line
            else:
                buf = line
    if buf:
        out.append(buf)
    return "\n\n".join(out)

def extract_native_text_blocks(page: fitz.Page):
    blocks = page.get_text("blocks", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
    lines = []
    for x0, y0, x1, y1, text, *_ in blocks:
        if not text or text.isspace():
            continue
        for raw_line in text.splitlines():
            line = normalize_line(raw_line)
            if line:
                fontsize = max(8.0, (y1 - y0) / max(1, text.count("\n") + 1))
                lines.append((line, (x0, y0, x1, y1), fontsize))
    lines.sort(key=lambda it: (it[1][1], it[1][0]))
    return lines

def extract_text_with_easyocr(page: fitz.Page, lang_codes: List[str]):
    pix = page.get_pixmap(dpi=220)
    img = Image.open(io.BytesIO(pix.tobytes()))
    arr = np.array(img)
    reader = get_easyocr_reader(lang_codes)
    results = reader.readtext(arr, detail=1, paragraph=False)
    lines = []
    for bbox, raw, _ in results:
        line = normalize_line(raw)
        if not line:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        fontsize = max(8.0, (y1 - y0))
        lines.append((line, (x0, y0, x1, y1), fontsize))
    lines.sort(key=lambda it: (it[1][1], it[1][0]))
    return lines

def clean_document_lines(pages_lines_with_geo):
    simple_pages = [[ln for (ln, _, _) in page] for page in pages_lines_with_geo]
    top_repeats, bot_repeats = gather_repeating_headers_footers(simple_pages)
    cleaned_all: List[str] = []
    for page in pages_lines_with_geo:
        if not page:
            continue
        page_heights = [bbox[3] for (_, bbox, _) in page]
        page_height = max(page_heights) if page_heights else 1000
        bottom_cut = page_height * 0.15
        page_clean: List[str] = []
        for line, bbox, fontsize in page:
            txt = line
            if txt in top_repeats or txt in bot_repeats:
                continue
            if is_probable_page_number(txt):
                continue
            txt = re.sub(r"(?<!\w)(\d{1,3}|[ivxlcdm]{1,4})\s*$", "", txt, flags=re.I).strip()
            if not txt:
                continue
            if len(txt) <= 2:
                continue
            y1 = bbox[3]
            if y1 >= page_height - bottom_cut:
                if re.match(r"^\s*\(?\d{1,3}[a-zA-Z\)]?\s+.*", txt) or fontsize < 10:
                    continue
            page_clean.append(txt)
        dedup = []
        for t in page_clean:
            if not dedup or dedup[-1] != t:
                dedup.append(t)
        cleaned_all.extend(dedup + [""])
    return stitch_paragraphs(cleaned_all)

st.set_page_config(page_title="PDF → Clean Text (OCR)", page_icon="🧹", layout="centered")
st.title("PDF → Clean Text (with OCR)")
st.write("Upload PDFs to get only the main body text — no images, charts, page numbers, repeated headers/footers, or footnotes.")

with st.expander("OCR & Extraction Options"):
    use_ocr_fallback = st.checkbox("Use OCR fallback if a page has little/no native text", value=True)
    ocr_langs = st.text_input("OCR languages (comma-separated EasyOCR codes)", value="en")
    header_footer_threshold = st.slider("Header/Footer repetition threshold", 0.3, 0.9, 0.5, 0.1)

uploaded = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

if uploaded:
    for file in uploaded:
        st.subheader(file.name)
        try:
            data = file.read()
            doc = fitz.open(stream=data, filetype="pdf")
            pages_lines_with_geo = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                native = extract_native_text_blocks(page)
                native_text_len = sum(len(t) for (t, _, _) in native)
                if use_ocr_fallback and native_text_len < 80:
                    try:
                        langs = [c.strip() for c in ocr_langs.split(",") if c.strip()]
                        ocr_lines = extract_text_with_easyocr(page, langs)
                        lines = ocr_lines if len(ocr_lines) > len(native) else native
                    except Exception as e:
                        st.info(f"OCR failed on page {i+1}: {e}. Using native text only.")
                        lines = native
                else:
                    lines = native
                pages_lines_with_geo.append(lines)
            cleaned_text = clean_document_lines(pages_lines_with_geo)
            txt_bytes = cleaned_text.encode("utf-8")
            out_name = re.sub(r"\.pdf$", "", file.name, flags=re.I) + "_clean.txt"
            st.download_button(
                label=f"Download cleaned text for {file.name}",
                data=txt_bytes,
                file_name=out_name,
                mime="text/plain"
            )
            with st.expander("Preview cleaned text"):
                st.text_area("", cleaned_text[:100000], height=300)
        except Exception as e:
            st.error(f"Failed to process {file.name}: {e}")
