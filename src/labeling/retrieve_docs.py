from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from src.utils.config import PROJECT_ROOT


VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "chroma"
COLLECTION_NAME = "support_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class SupportKnowledgeRetriever:
    """Retrieve relevant support knowledge base documents using semantic search."""

    def __init__(self, vector_store_dir: Path = VECTOR_STORE_DIR):
        print(f"Looking for vector store at: {vector_store_dir}", flush=True)
        print(f"Vector store exists: {vector_store_dir.exists()}", flush=True)

        if not vector_store_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_dir}. "
                "Run `python -m src.rag.build_vector_store` first."
            )

        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}", flush=True)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        print("Embedding model loaded successfully.", flush=True)

        print("Importing ChromaDB...", flush=True)
        import chromadb
        print("ChromaDB imported successfully.", flush=True)

        print("Connecting to Chroma persistent client...", flush=True)
        self.client = chromadb.PersistentClient(path=str(vector_store_dir))

        existing_collections = [
            collection.name for collection in self.client.list_collections()
        ]
        print(f"Available collections: {existing_collections}", flush=True)

        if COLLECTION_NAME not in existing_collections:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run `python -m src.rag.build_vector_store` first."
            )

        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        print("Retriever initialized successfully.", flush=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant knowledge base chunks for a query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_docs = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            retrieved_docs.append({
                "source": metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": round(float(distance), 4),
                "content": doc
            })

        return retrieved_docs


def main() -> None:
    print("Starting retrieval test...", flush=True)

    retriever = SupportKnowledgeRetriever()

    sample_queries = [
        "I was charged twice for my order and need help with the payment.",
        "I want to cancel my subscription.",
        "I cannot login to my account after resetting my password.",
        "I want to know if the product works with my device.",
        "The customer is angry and wants to escalate the issue."
    ]

    for query in sample_queries:
        print("\n" + "=" * 100, flush=True)
        print("Query:", query, flush=True)
        print("=" * 100, flush=True)

        docs = retriever.retrieve(query, top_k=2)

        if not docs:
            print("No documents retrieved.", flush=True)

        for doc in docs:
            print("Source:", doc["source"], flush=True)
            print("Distance:", doc["distance"], flush=True)
            print("Content Preview:", doc["content"][:400], flush=True)
            print("-" * 100, flush=True)

    print("Retrieval test completed.", flush=True)


if __name__ == "__main__":
    main()