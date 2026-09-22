from pathlib import Path
import time
import requests


def download_pdf(paper: dict, destination: Path, max_retries: int = 3) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 10_000:
        return destination

    for attempt in range(max_retries):
        try:
            # Stream the download to handle large PDFs efficiently
            response = requests.get(
                paper["pdf_url"],
                stream=True,
                timeout=60,
                headers={"User-Agent": "arxiv-research-agent/1.0"},
            )
            response.raise_for_status()

            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify it looks like a PDF
            with open(destination, "rb") as f:
                header = f.read(5)
            if not header.startswith(b"%PDF"):
                destination.unlink(missing_ok=True)
                raise RuntimeError("Downloaded content does not look like a PDF.")

            return destination

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Download failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    return destination
