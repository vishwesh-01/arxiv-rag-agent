import base64
import re
from pathlib import Path
import pymupdf as fitz
import pymupdf4llm
from langchain_core.messages import HumanMessage, SystemMessage
from llm.gemini import invoke_vision

def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Primary parser: PyMuPDF4LLM page chunks enhanced with Gemini Vision
    for visual candidate pages (figures, tables, equations, diagrams).
    """
    doc = fitz.open(pdf_path)
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        write_images=False,
    )

    visual_candidates = detect_visual_candidates(pdf_path)
    visual_pages = {c["page"] for c in visual_candidates}

    output = []
    for index, item in enumerate(chunks):
        text = (item.get("text") or "").strip()
        page = int(item.get("metadata", {}).get("page", index + 1))
        if text:
            output.append({
                "page": page,
                "text": text,
                "type": "text",
            })

        # Process page visual content if signaled
        if page in visual_pages:
            visual_items = process_page_visuals(doc[page - 1], page)
            for item in visual_items:
                output.append(item)

    doc.close()

    # Basic quality signal for scanned/broken PDFs.
    total_chars = sum(len(x["text"]) for x in output)
    if total_chars < 1500:
        output = fallback_plain_text(pdf_path)

    return output

def process_page_visuals(page: fitz.Page, page_num: int) -> list[dict]:
    """
    Renders visual pages to high-resolution images and uses Gemini Vision
    to parse tables, figures, diagrams, and LaTeX equations.
    """
    try:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "Analyze this academic paper page image:\n"
            "1. Convert any tables into clean Markdown tables.\n"
            "2. Extract mathematical equations into LaTeX syntax ($$ ... $$).\n"
            "3. Describe any figures, diagrams, architectural charts, or plots in detail.\n"
            "Format output with clear headers like [TABLE], [EQUATION], or [FIGURE]."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ]
        )

        response = invoke_vision([
            SystemMessage(content="You are a scientific document visual parser."),
            message,
        ])
        
        raw_content = response.content
        if isinstance(raw_content, list):
            parts = []
            for item in raw_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            content = "\n".join(parts).strip()
        else:
            content = str(raw_content or "").strip()

        if not content:
            return []

        results = []
        if "[TABLE]" in content or "|---" in content or "Table" in content:
            results.append({"page": page_num, "text": content, "type": "table"})
        elif "[FIGURE]" in content or "Figure" in content:
            results.append({"page": page_num, "text": content, "type": "figure"})
        elif "[EQUATION]" in content or "$$" in content:
            results.append({"page": page_num, "text": content, "type": "equation"})
        else:
            results.append({"page": page_num, "text": content, "type": "visual_summary"})

        return results
    except Exception as e:
        print(f"Visual processing failed for page {page_num}: {e}")
        return []

def fallback_plain_text(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    result = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            result.append({
                "page": i + 1,
                "text": text,
                "type": "text",
            })
    doc.close()
    return result

def detect_visual_candidates(pdf_path: Path) -> list[dict]:
    """
    Conservative local detector. It marks pages for visual analysis
    when the page has image/graphics content and nearby text contains
    figure/table/algorithm/architecture signals.
    """
    doc = fitz.open(pdf_path)
    candidates = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        lower = text.lower()

        caption_signal = bool(re.search(
            r"\b(fig(?:ure)?\.?|table|algorithm|architecture|equation|formula)\s*\d*\b",
            lower,
        ))
        images = page.get_images(full=True)
        drawings = page.get_drawings()

        if caption_signal and (images or len(drawings) >= 5):
            candidates.append({
                "page": i + 1,
                "reason": "caption + image/vector signal",
            })

    doc.close()
    return candidates

