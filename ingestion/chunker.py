from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages: list[dict], arxiv_id: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for page in pages:
        block_type = page.get("type", "text")
        page_num = page.get("page", 1)
        text = page.get("text", "").strip()

        if not text:
            continue

        # Keep specialized visual/table/figure/equation chunks intact
        if block_type in ("table", "figure", "equation", "visual_summary"):
            documents.append(Document(
                page_content=text,
                metadata={
                    "arxiv_id": arxiv_id,
                    "page": page_num,
                    "type": block_type,
                }
            ))
        else:
            docs = splitter.create_documents(
                [text],
                metadatas=[{
                    "arxiv_id": arxiv_id,
                    "page": page_num,
                    "type": block_type,
                }],
            )
            documents.extend(docs)

    return documents

