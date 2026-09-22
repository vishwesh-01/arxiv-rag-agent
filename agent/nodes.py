import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from llm.gemini import invoke_chat
from models.schemas import PaperBriefing
from fetch_arxiv.client import normalize_arxiv_id, get_paper, search_papers
from ingestion.downloader import download_pdf
from ingestion.parser import parse_pdf
from ingestion.chunker import chunk_pages
from retrieval.chroma_store import (
    collection_name,
    exists,
    create,
    load,
    count,
)

PAPERS_FILE = Path("papers.json")
PAPERS_DIR = Path("data/papers")

def load_registry():
    if not PAPERS_FILE.exists():
        return {}
    try:
        return json.loads(PAPERS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def save_registry(data):
    PAPERS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def understand_query(state):
    value = state["user_input"].strip()
    arxiv_id = normalize_arxiv_id(value)

    if arxiv_id:
        return {
            "input_type": "paper",
            "arxiv_id": arxiv_id,
            "query": None,
        }

    return {
        "input_type": "topic",
        "query": value,
        "arxiv_id": None,
    }

def retrieve_arxiv(state):
    if state["input_type"] == "paper":
        paper = get_paper(state["arxiv_id"])
        return {"selected_paper": paper, "candidates": [paper]}

    candidates = search_papers(state["query"], max_results=8)
    return {"candidates": candidates}

def select_paper(state):
    if state.get("selected_paper"):
        return {}

    candidates = state.get("candidates", [])
    if not candidates:
        return {"error": "arXiv returned no candidate papers."}

    # The CLI selects before this graph is resumed in interactive mode.
    # Default to the first relevance-ranked result for programmatic use.
    return {"selected_paper": candidates[0]}

def ensure_indexed(state):
    paper = state["selected_paper"]
    arxiv_id = paper["arxiv_id"]
    registry = load_registry()

    cached = (
        arxiv_id in registry
        and registry[arxiv_id].get("indexed") is True
        and exists(arxiv_id)
        and count(arxiv_id) > 0
    )

    if cached:
        return {
            "indexed": True,
            "collection_name": collection_name(arxiv_id),
        }

    pdf_path = PAPERS_DIR / f"{arxiv_id}.pdf"
    pdf_path = download_pdf(paper, pdf_path)

    pages = parse_pdf(pdf_path)
    if not pages:
        raise RuntimeError("PDF parsing produced no usable text.")

    docs = chunk_pages(pages, arxiv_id)
    create(arxiv_id, docs)

    registry[arxiv_id] = {
        **paper,
        "indexed": True,
        "collection": collection_name(arxiv_id),
        "chunk_count": len(docs),
        "pdf_path": str(pdf_path),
    }
    save_registry(registry)

    return {
        "indexed": False,
        "collection_name": collection_name(arxiv_id),
        "pdf_path": str(pdf_path),
        "parsed_documents": pages,
    }

def generate_briefing(state):
    paper = state["selected_paper"]
    store = load(paper["arxiv_id"])

    docs = store.similarity_search(
        "abstract problem method approach experiments results limitations",
        k=10,
    )

    context = "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page')}] {d.page_content}"
        for d in docs
    )

    prompt = f"""
Create an accurate executive briefing for this arXiv paper.

Metadata:
{json.dumps(paper, indent=2)}

Use the retrieved paper excerpts below. Do not invent facts.
If a limitation is not explicitly stated, infer only cautiously from
the paper's evidence and label it as an observed limitation.

Retrieved excerpts:
{context}
"""

    response = invoke_chat([
        SystemMessage(content="You are a careful research analyst."),
        HumanMessage(content=prompt),
    ])

    structured = invoke_chat([
        SystemMessage(content=(
            "Convert the following draft into the requested JSON schema. "
            "Return only valid JSON."
        )),
        HumanMessage(content=response.content),
    ])

    try:
        briefing = PaperBriefing.model_validate_json(
            structured.content
        ).model_dump()
    except Exception:
        # Keep a usable fallback rather than failing the whole graph.
        briefing = {
            "title": paper["title"],
            "authors": paper["authors"],
            "arxiv_id": paper["arxiv_id"],
            "publish_date": paper["published"],
            "link": paper["abs_url"],
            "why_it_matters": response.content,
            "problem_statement": "",
            "method": [],
            "key_results": [],
            "limitations": ["Structured extraction failed; review the source text."],
            "follow_up_questions": [],
        }

    return {"briefing": briefing}

def retrieve_for_qa(state):
    paper = state["selected_paper"]
    store = load(paper["arxiv_id"])

    docs = store.similarity_search(
        state["question"],
        k=8,
    )

    retrieved = []
    for d in docs:
        retrieved.append({
            "content": d.page_content,
            "page": d.metadata.get("page"),
            "type": d.metadata.get("type", "text"),
        })

    return {"retrieved_docs": retrieved}

def answer_qa(state):
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            "answer": "I couldn't find enough information about that in the paper.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[Page {d['page']} ({d.get('type', 'text')})]\n{d['content']}"
        for d in docs
    )

    prompt = f"""
Answer the question using ONLY the supplied excerpts from the selected
arXiv paper.

Rules:
- Do not use outside knowledge.
- If the answer is not supported, say:
  "I couldn't find enough information about that in the paper."
- Cite page numbers for factual claims.
- Do not pretend an inference is explicitly stated.

Question:
{state['question']}

Paper excerpts:
{context}
"""

    response = invoke_chat([
        SystemMessage(content="You are a grounded research-paper QA assistant."),
        HumanMessage(content=prompt),
    ])

    sources = [
        {"page": d["page"], "type": d["type"]}
        for d in docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
