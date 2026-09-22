# Autonomous Multimodal arXiv Paper Digest & QA Agent

An autonomous local CLI research assistant for arXiv scientific papers built with **LangGraph**, **PyMuPDF**, **ChromaDB**, and **Gemini Vision**.

It accepts:
- A natural-language research topic (e.g., `"transformer attention mechanism"`)
- An arXiv ID (e.g., `"1706.03762"`)
- An arXiv URL (e.g., `"https://arxiv.org/abs/1706.03762"`)

It searches/fetches the paper, parses text and visual components (diagrams, architecture charts, visual tables, LaTeX equations), creates a structured executive briefing, and launches an interactive grounded RAG QA session.

---

## State Graph Architecture

The workflow is orchestrated using **LangGraph** with explicit state transitions.

```text
               +-----------------------+
               |     User Input        |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |   understand_query    |
               +-----------+-----------+
                           |
           +---------------+---------------+
           |                               |
 (arXiv ID or URL)                   (Topic Query)
           |                               |
           v                               v
 +-------------------+           +-------------------+
 |  retrieve_arxiv   |           |  retrieve_arxiv   |
 |  (Get single paper|           |  (Search 8 papers)|
 +---------+---------+           +---------+---------+
           |                               |
           |                               v
           |                     +-------------------+
           |                     |   select_paper    |
           |                     +---------+---------+
           |                               |
           +---------------+---------------+
                           |
                           v
               +-----------------------+
               |    ensure_indexed     |
               | (Visual PDF Parser &  |
               | Chroma Vector Store)  |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |   generate_briefing   |
               |  (Structured Brief)   |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               |     QA Loop           |
               | retrieve_for_qa       |
               | answer_qa             |
               +-----------------------+
```

### State Shape (`AgentState`)

```python
class AgentState(TypedDict):
    user_input: str                       # Raw query, ID, or URL
    input_type: Optional[str]             # "paper" or "topic"
    query: Optional[str]                  # Search query string if topic
    arxiv_id: Optional[str]               # Normalized arXiv ID
    candidates: list[dict]                # Candidate arXiv paper metadata list
    selected_paper: Optional[dict]        # Active selected paper metadata
    indexed: bool                         # True if Chroma collection is loaded from cache
    collection_name: Optional[str]        # Chroma collection identifier
    parsed_documents: list[dict]          # Output blocks from PyMuPDF & Gemini Vision
    briefing: Optional[dict]              # Executive briefing object matching PaperBriefing schema
    question: Optional[str]               # User QA question
    retrieved_docs: list[dict]            # Retrieved context chunks for QA
    answer: Optional[str]                 # Grounded response string
    sources: list[dict]                   # List of source citations with page numbers & block types
    error: Optional[str]                  # Error tracking message
```

---

## Multimodal Vision & Parsing Pipeline

Scientific papers convey critical ideas through architectural diagrams, plots, data tables, and mathematical formulas. Standard PDF text extractors frequently mangle or lose these visual elements.

```text
                      PDF File (data/papers/<arxiv_id>.pdf)
                                      |
                                      v
                             PyMuPDF4LLM Parsing
                                      |
                                      v
                        detect_visual_candidates()
            (Scans for captions, images, vector drawing clusters)
                                      |
                   +------------------+------------------+
                   |                                     |
           (Text-only Pages)                    (Visual Candidate Pages)
                   |                                     |
                   v                                     v
            Raw Text Chunks                     PyMuPDF Page Render (150 DPI)
                   |                                     |
                   |                                     v
                   |                            Gemini Vision LLM
                   |                       (gemma-4-31b / gemini-vision)
                   |                                     |
                   |                 +-------------------+-------------------+
                   |                 |                   |                   |
                   |                 v                   v                   v
                   |          Markdown Tables    LaTeX Equations    Figure Descriptions
                   |           (type: table)     (type: equation)     (type: figure)
                   +-----------------+-------------------+-------------------+
                                     |
                                     v
                              chunk_pages()
              (Preserves table/figure/equation chunks intact)
                                     |
                                     v
                               ChromaDB
```

1. **Selective Visual Candidate Detection**: `detect_visual_candidates()` scans PDF pages for figure/table/algorithm/equation caption signals and vector drawing density.
2. **Dedicated Multimodal Model**: Pages with visual signals are rendered at high resolution (150 DPI) and processed by `GEMINI_VISION_MODEL` (`invoke_vision()`).
3. **Structured Content Extraction**:
   - **Tables**: Converted into clean Markdown tables (`type: "table"`).
   - **Figures & Diagrams**: Extracted with full structural descriptions of flow charts, plots, and network architectures (`type: "figure"`).
   - **Math & Equations**: Extracted using clean LaTeX notation (`$$ ... $$`, `type: "equation"`).
4. **Type-Aware Chunking**: Visual blocks (`table`, `figure`, `equation`) are kept intact during chunking to preserve complete contextual boundaries.

---

## Setup & Running Locally

### Prerequisites
- Python 3.11+
- Free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/vishwesh-01/arxiv-rag-agent.git
cd arxiv-rag-agent

# Create a virtual environment
python -m venv .venv
```

Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and set your API key and model choices:

```env
GEMINI_API_KEY=your_google_ai_studio_api_key

# Model Configurations
GEMINI_CHAT_MODEL=gemini-3.1-flash
GEMINI_VISION_MODEL=gemma-3-27b-it
GEMINI_EMBED_MODEL=gemini-embedding-001

# Rate Limiter & Throttling
GEMINI_MIN_REQUEST_INTERVAL=4
```

### 4. Run the Agent

```bash
python app.py
```

---

## Example Run

### Input & Paper Selection

```text
==================================================
  Autonomous arXiv Paper Digest & QA Agent
==================================================

Enter an arXiv ID (e.g. 1706.03762), an arXiv URL,
or a research topic to search for.
Type 'exit' to quit.

> 1706.03762
```

### Output 1: Executive Briefing

```json
{
  "title": "Attention Is All You Need",
  "authors": [
    "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
    "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"
  ],
  "arxiv_id": "1706.03762",
  "publish_date": "2017-06-12T17:57:34Z",
  "link": "http://arxiv.org/abs/1706.03762v7",
  "why_it_matters": "Replaces recurrent and convolutional layers with self-attention mechanisms, setting new benchmarks in machine translation with drastically reduced training time.",
  "problem_statement": "Dominant sequence transduction models rely on complex recurrent or convolutional networks, which limits parallelization across long sequences.",
  "method": [
    "Transformer encoder-decoder architecture using stacked multi-head self-attention.",
    "Positional encodings added to input embeddings to inject sequence order.",
    "Scaled Dot-Product Attention computed as Softmax(Q K^T / sqrt(d_k)) V."
  ],
  "key_results": [
    "Achieves 28.4 BLEU on English-to-German translation (improving by 2.0 BLEU).",
    "Reaches 41.8 BLEU on English-to-French translation with 3.5 days of training on 8 GPUs."
  ],
  "limitations": [
    "Quadratic computational complexity with respect to sequence length in self-attention.",
    "Observed sensitivity to hyperparameter tuning and learning rate warm-up schedules."
  ],
  "follow_up_questions": [
    "How does positional encoding perform relative to learned positional embeddings?",
    "What is the impact of multi-head attention versus single-head attention?"
  ]
}
```

### Output 2: Interactive QA Session

```text
--------------------------------------------------
  Paper indexed and executive briefing created.
  Entering grounded QA session.
  Type 'back' to choose another paper or 'exit' to quit.
--------------------------------------------------

Question: Can you describe the architectural diagram of the Transformer?

Assistant:
According to Figure 1 on page 3, the Transformer consists of an Encoder (left) and a Decoder (right), stacked N = 6 times.

- **Encoder**: Contains a multi-head self-attention mechanism followed by a position-wise feed-forward network, with residual connections and layer normalization around each sub-layer.
- **Decoder**: Contains a multi-head self-attention mechanism, a multi-head cross-attention mechanism over the encoder outputs, and position-wise feed-forward networks.

Sources:
  - Page 3 (figure)
  - Page 4 (figure)

--------------------------------------------------

Question: What BLEU scores did the model achieve on English to German and English to French?

Assistant:
As shown in Table 2 on page 8:
- **English-to-German (newstest2014)**: 28.4 BLEU (outperforming previous state-of-the-art models including Ensemble models).
- **English-to-French (newstest2014)**: 41.8 BLEU.

Sources:
  - Page 8 (table)

--------------------------------------------------

Question: What is the exact formula for Scaled Dot-Product Attention?

Assistant:
The formula for Scaled Dot-Product Attention is:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$

where $Q$ is the Query matrix, $K$ is the Key matrix, $V$ is the Value matrix, and $d_k$ is the dimensionality of the keys.

Sources:
  - Page 4 (equation)
  - Page 4 (text)
```

---

## Design Decisions & Tradeoffs

### 1. Selective Multimodal Vision Processing vs. Blind Page-by-Page Vision
- **Decision**: Use `detect_visual_candidates()` to flag candidate pages containing images, charts, tables, or vector drawings, and render only those pages for vision LLM analysis.
- **Tradeoff**: Saves significant API quota, cost, and latency compared to sending every PDF page to a vision model, while preserving rich descriptions for complex diagrams and tables.

### 2. Dedicated Vision Model Decoupling (`GEMINI_VISION_MODEL`)
- **Decision**: Decoupled vision execution (`invoke_vision()`) from text QA (`invoke_chat()`).
- **Tradeoff**: Allows users to configure lightweight/specialized vision models (e.g., `gemma-4-31b-it` or `gemma-3-27b-it`) for document parsing while using faster chat models for QA logic.

### 3. Type-Aware Block Chunking vs. Standard Character Splitting
- **Decision**: `chunk_pages()` identifies block metadata types (`table`, `figure`, `equation`) and preserves them as whole chunks rather than breaking Markdown tables or LaTeX blocks across character split points.
- **Tradeoff**: Slightly larger individual chunk sizes for visual blocks, but guarantees complete context preservation during vector retrieval.

### 4. Per-Paper Isolated Vector Collections in ChromaDB
- **Decision**: Store embeddings in paper-isolated Chroma collections (`paper_1706_03762`).
- **Tradeoff**: Prevents cross-paper chunk pollution and speeds up query retrieval, though multi-paper comparative RAG requires querying across collections.

---

## Known Limitations & Future Improvements

1. **Bounding-Box Cropping**: Currently, candidate pages are rendered as full pages at 150 DPI. Cropping specific bounding boxes for individual figures will reduce payload size further.
2. **Hybrid Retrieval**: Combining dense vector embeddings with sparse keyword search (BM25) would improve retrieval on rare mathematical symbols or hyperparameter names.
3. **Local Reranking**: Incorporating a cross-encoder reranker (e.g., `bge-reranker`) prior to passing context to the QA model.
