import base64
import concurrent.futures
import re
from pathlib import Path
import pymupdf as fitz
import pymupdf4llm
from langchain_core.messages import HumanMessage, SystemMessage
from llm.gemini import invoke_vision

def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Primary parser: PyMuPDF4LLM page chunks enhanced with parallel Gemini Vision
    for visual candidate pages (figures, tables, equations, diagrams).
    """
    doc = fitz.open(pdf_path)
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        write_images=False,
    )

    visual_candidates = detect_visual_candidates(pdf_path)
    visual_pages_sorted = sorted(list({c["page"] for c in visual_candidates}))

    # Extract base64 image strings safely on main thread before threading
    visual_tasks = []
    for p_num in visual_pages_sorted:
        page = doc[p_num - 1]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        visual_tasks.append((p_num, img_b64))

    doc.close()

    # Process visual candidate LLM API calls concurrently
    page_visual_map = {}
    if visual_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_page = {
                executor.submit(process_page_visual_b64, img_b64, p_num): p_num
                for p_num, img_b64 in visual_tasks
            }
            for future in concurrent.futures.as_completed(future_to_page):
                p_num = future_to_page[future]
                try:
                    page_visual_map[p_num] = future.result()
                except Exception as exc:
                    print(f"Parallel visual processing failed for page {p_num}: {exc}")
                    page_visual_map[p_num] = []

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

        # Attach parallel visual items for this page
        if page in page_visual_map:
            for v_item in page_visual_map[page]:
                output.append(v_item)

    # Basic quality signal for scanned/broken PDFs.
    total_chars = sum(len(x["text"]) for x in output)
    if total_chars < 1500:
        output = fallback_plain_text(pdf_path)

    return output

def process_page_visual_b64(img_b64: str, page_num: int) -> list[dict]:
    """
    Sends base64 image string to Gemini Vision to parse tables, figures, diagrams, and LaTeX equations.
    """
    try:
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

