from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from src.utils.config import PROJECT_ROOT


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "chroma"
COLLECTION_NAME = "support_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_markdown_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> List[dict]:
    """Load markdown documents from the knowledge base directory."""
    print("Loading markdown documents...", flush=True)
    print(f"Knowledge base directory: {kb_dir}", flush=True)
    print(f"Knowledge base exists: {kb_dir.exists()}", flush=True)

    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    markdown_files = list(kb_dir.glob("*.md"))
    print(f"Markdown files found: {len(markdown_files)}", flush=True)

    documents = []

    for file_path in markdown_files:
        print(f"Reading: {file_path.name}", flush=True)
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    if not documents:
        raise ValueError("No markdown documents found.")

    return documents


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def build_chunks(documents: List[dict]) -> tuple[list, list, list]:
    """Create IDs, text chunks, and metadata."""
    ids = []
    texts = []
    metadatas = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        print(f"{doc['source']} split into {len(chunks)} chunks.", flush=True)

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}_chunk_{chunk_index}"

            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "source": doc["source"],
                "chunk_index": chunk_index
            })

    print(f"Total chunks created: {len(texts)}", flush=True)

    if not texts:
        raise ValueError("No chunks were created.")

    return ids, texts, metadatas


def build_vector_store() -> None:
    """Build and persist a Chroma vector store."""
    print("Starting vector store build...", flush=True)

    documents = load_markdown_documents()
    ids, texts, metadatas = build_chunks(documents)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Vector store directory: {VECTOR_STORE_DIR}", flush=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    print("Embedding model loaded successfully.", flush=True)

    print("Creating embeddings...", flush=True)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).tolist()
    print("Embeddings created successfully.", flush=True)

    print("Importing ChromaDB...", flush=True)
    import chromadb
    print("ChromaDB imported successfully.", flush=True)

    print("Connecting to Chroma persistent client...", flush=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    existing_collections = [collection.name for collection in client.list_collections()]
    print(f"Existing collections: {existing_collections}", flush=True)

    if COLLECTION_NAME in existing_collections:
        print(f"Deleting existing collection: {COLLECTION_NAME}", flush=True)
        client.delete_collection(name=COLLECTION_NAME)

    print(f"Creating collection: {COLLECTION_NAME}", flush=True)
    collection = client.create_collection(name=COLLECTION_NAME)

    print("Adding documents to Chroma...", flush=True)
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("Vector store built successfully.", flush=True)
    print(f"Persisted at: {VECTOR_STORE_DIR}", flush=True)
    print(f"Collection name: {COLLECTION_NAME}", flush=True)


def main() -> None:
    build_vector_store()


if __name__ == "__main__":
    main()