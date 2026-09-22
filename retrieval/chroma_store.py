import os
from pathlib import Path
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from llm.gemini import embedding_model

CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")

def collection_name(arxiv_id: str) -> str:
    safe = arxiv_id.replace(".", "_").replace("-", "_").replace("/", "_")
    return f"paper_{safe}"

def get_client():
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

def exists(arxiv_id: str) -> bool:
    name = collection_name(arxiv_id)
    try:
        get_client().get_collection(name)
        return True
    except Exception:
        return False

def count(arxiv_id: str) -> int:
    try:
        return get_client().get_collection(collection_name(arxiv_id)).count()
    except Exception:
        return 0

def load(arxiv_id: str) -> Chroma:
    return Chroma(
        client=get_client(),
        collection_name=collection_name(arxiv_id),
        embedding_function=embedding_model,
    )

def create(arxiv_id: str, documents: list[Document]) -> Chroma:
    if not documents:
        raise ValueError("No parsed documents were produced.")

    # Chroma/LangChain handles batching internally.
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=get_client(),
        collection_name=collection_name(arxiv_id),
    )
