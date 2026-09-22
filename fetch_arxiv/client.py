import re
import time
import logging

import arxiv
import requests

from models.schemas import PaperMetadata

logger = logging.getLogger(__name__)

ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
    re.IGNORECASE,
)

# Reuse a single Client instance across calls.
# delay_seconds=3 respects arXiv's rate limit policy.
client = arxiv.Client(
    page_size=1,
    delay_seconds=3,
    num_retries=5,
)


def normalize_arxiv_id(value: str) -> str | None:
    value = value.strip()

    match = ID_RE.search(value)

    if not match:
        return None

    return match.group(1)


def _metadata(paper) -> dict:
    arxiv_id = paper.entry_id.split("/")[-1]

    return PaperMetadata(
        title=paper.title.strip(),
        authors=[author.name for author in paper.authors],
        arxiv_id=arxiv_id,
        published=paper.published.date().isoformat(),
        updated=paper.updated.date().isoformat() if paper.updated else None,
        summary=paper.summary.strip(),
        categories=list(paper.categories),
        pdf_url=paper.pdf_url,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    ).model_dump()


def _fetch_with_retry(search, max_retries: int = 3):
    """Wrap client.results() with manual retry on 429 / connection errors."""
    for attempt in range(max_retries):
        try:
            results = client.results(search)
            paper = next(results)
            return paper
        except StopIteration:
            return None
        except arxiv.HTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                logger.warning(f"arXiv 429, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                print(f"  arXiv rate-limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"arXiv error: {e}, retrying in {wait}s")
                print(f"  arXiv connection error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return None


def get_paper(arxiv_id: str) -> dict:
    """Fetch a single paper by its arXiv ID."""
    search = arxiv.Search(id_list=[arxiv_id])

    paper = _fetch_with_retry(search)
    if paper is None:
        raise ValueError(f"No arXiv paper found for {arxiv_id}")

    return _metadata(paper)


def search_papers(query: str, max_results: int = 8) -> list[dict]:
    """Search arXiv by topic query."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    try:
        for paper in client.results(search):
            papers.append(_metadata(paper))
    except Exception as e:
        logger.warning(f"Search error: {e}")
        if not papers:
            raise

    return papers