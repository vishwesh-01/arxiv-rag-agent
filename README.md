# Autonomous arXiv Paper Digest & QA Agent

A local CLI research assistant for arXiv papers.

It accepts either:

- a natural-language research topic
- an arXiv ID
- an arXiv URL

It searches/fetches the paper, parses it locally, creates a structured executive briefing, and opens a grounded RAG QA loop.

## Architecture

```text
User
 |
 v
Query Understanding
 |
 +---- arXiv ID/URL --------+
 |                          |
 +---- Topic -> Search -> Rank
                            |
                            v
                       Paper Registry
                            |
                    already indexed?
                       /          \
                     yes           no
                      |             |
                      |        Download PDF
                      |             |
                      |        PyMuPDF4LLM
                      |             |
                      |        Chunk + metadata
                      |             |
                      |        Gemini embeddings
                      |             |
                      +------> Chroma
                                  |
                                  v
                           Executive Briefing
                                  |
                                  v
                              QA session
                                  |
                                  v
                            Chroma retrieval
                                  |
                                  v
                           Grounded Gemini
```

## Why this PDF design?

The first implementation rendered every PDF page and sent every page to a vision model. That is expensive and can incorrectly treat author photos, logos, or decorative images as research figures.

This version uses PyMuPDF4LLM for local, layout-aware PDF extraction and only uses LLM calls for summarization/QA. A conservative local detector is provided for future selective figure/table vision processing.

## Cache design

Each paper is identified by its arXiv ID.

Example:

```text
1706.03762
    |
    +-- data/papers/1706.03762.pdf
    |
    +-- Chroma collection: paper_1706_03762
    |
    +-- papers.json
```

If the same paper is selected again:

1. The PDF is not downloaded again.
2. The PDF is not parsed again.
3. The chunks are not embedded again.
4. The existing Chroma collection is loaded.

## Embeddings

One embedding model is used consistently:

```text
gemini-embedding-2-preview
```

Each paper has an isolated Chroma collection. "Different embeddings per paper" therefore means separate vector data per paper, not a different embedding model instance.

## Gemini free-tier handling

The application intentionally limits API usage:

- PDF parsing is local.
- No page-by-page Gemini vision calls.
- Embeddings are sent through LangChain's batch embedding operation.
- A client-side minimum request interval is configurable.
- Rate-limit/quota errors use exponential backoff.
- Cached papers do not require re-embedding.

The exact free-tier quotas vary by model and Google project and can change. Check Google's Gemini API rate-limit documentation before testing.

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Copy:

```text
.env.example -> .env
```

Set:

```env
GEMINI_API_KEY=your_key
```

Run:

```bash
python app.py
```

## Example

```text
Research topic or arXiv ID/URL:

> retrieval augmented generation for scientific papers
```

The agent returns candidate papers. Select one.

It then produces:

- title
- authors
- arXiv ID
- date
- link
- why the paper matters
- problem
- method
- key results
- limitations
- follow-up questions

Then:

```text
QA mode

You:
What problem does the paper solve?

Assistant:
...

Sources: page 2 (text), page 4 (text)
```

## Failure handling

### No search results

The agent reports that arXiv returned no candidates.

### Bad PDF

PyMuPDF4LLM is attempted first. A plain PyMuPDF text fallback is used if extracted text is suspiciously small.

### Gemini rate limit

The rate limiter waits and retries with exponential backoff. The cached PDF/Chroma state is preserved.

### Re-running a paper

The existing Chroma collection is reused.

## Design decisions & tradeoffs

### LangGraph

The assessment explicitly asks for an identifiable stateful graph. LangGraph makes the nodes and shared state explicit.

### PyMuPDF4LLM

The system does not assume that every page image is meaningful research content. Local parsing is cheaper and more deterministic than asking a multimodal model to inspect every page.

### Chroma

Chroma is local and simple for an assessment project. A collection is isolated per arXiv paper, which makes cache lookup straightforward.

### Gemini

Gemini is used because it has a practical free-tier path for the assessment. The API is wrapped with throttling and retry behavior.

### Grounding

QA answers are generated from retrieved paper chunks only. The prompt explicitly tells the model to refuse when the retrieved paper context does not support an answer.

## Known limitations

- Figure/table visual understanding is intentionally conservative in this first implementation.
- Topic ranking currently relies on arXiv relevance ordering rather than a separate cross-encoder.
- Retrieval uses vector similarity without a dedicated reranker.
- Sessions are stored in JSON because multi-user persistence is out of scope.
- The system is designed for local CLI use, not production deployment.

## Suggested next improvements

1. Add section-aware chunking.
2. Add BM25 + vector hybrid retrieval.
3. Add a local reranker.
4. Extract figure/table captions and selectively send only those regions to Gemini vision.
5. Add citation-aware answer formatting.
6. Add tests around PDF parsing and cache reuse.
