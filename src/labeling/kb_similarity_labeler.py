from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.labeling.label_taxonomy import TICKET_TYPES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


REFERENCE_FILES = {
    "Billing inquiry": "billing_inquiry.md",
    "Cancellation request": "cancellation_request.md",
    "Product inquiry": "product_inquiry.md",
    "Refund request": "refund_request.md",
    "Technical issue": "technical_issue.md",
}


@dataclass
class KBSimilarityPrediction:
    suggested_ticket_type: Optional[str]
    confidence: float
    similarity_scores: Dict[str, float]
    top_reference_file: Optional[str]
    explanation: str

    def to_dict(self) -> Dict:
        return asdict(self)


class KBSimilarityLabeler:
    """
    Suggest a ticket type by comparing ticket text embeddings with
    ticket-type reference documents.
    """

    def __init__(
        self,
        knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.knowledge_base_dir = knowledge_base_dir
        self.model_name = model_name

        self.reference_documents = self._load_reference_documents()

        print(f"Loading embedding model: {self.model_name}", flush=True)
        self.model = SentenceTransformer(self.model_name, device="cpu")
        print("Embedding model loaded.", flush=True)

        self.reference_embeddings = self._build_reference_embeddings()

    def _load_reference_documents(self) -> Dict[str, str]:
        documents = {}

        for ticket_type, filename in REFERENCE_FILES.items():
            file_path = self.knowledge_base_dir / filename

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Reference file not found for '{ticket_type}': {file_path}"
                )

            documents[ticket_type] = file_path.read_text(encoding="utf-8")

        return documents

    def _build_reference_embeddings(self) -> Dict[str, np.ndarray]:
        ticket_types = list(self.reference_documents.keys())
        texts = [
            self.reference_documents[ticket_type]
            for ticket_type in ticket_types
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return {
            ticket_type: embedding
            for ticket_type, embedding in zip(ticket_types, embeddings)
        }

    def predict(self, ticket_text: str) -> KBSimilarityPrediction:
        if not ticket_text or not ticket_text.strip():
            return KBSimilarityPrediction(
                suggested_ticket_type=None,
                confidence=0.0,
                similarity_scores={
                    ticket_type: 0.0
                    for ticket_type in TICKET_TYPES
                },
                top_reference_file=None,
                explanation="No ticket text was provided.",
            )

        ticket_embedding = self.model.encode(
            [ticket_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        similarity_scores = {}

        for ticket_type, reference_embedding in self.reference_embeddings.items():
            similarity = float(
                np.dot(ticket_embedding, reference_embedding)
            )

            similarity_scores[ticket_type] = round(similarity, 4)

        ranked_types = sorted(
            similarity_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_type, top_score = ranked_types[0]
        second_score = ranked_types[1][1] if len(ranked_types) > 1 else 0.0

        margin = max(top_score - second_score, 0.0)

        confidence = self._calculate_confidence(
            top_score=top_score,
            margin=margin,
        )

        top_reference_file = REFERENCE_FILES[top_type]

        explanation = (
            f"Suggested '{top_type}' because it had the highest semantic "
            f"similarity score ({top_score:.4f}) against "
            f"'{top_reference_file}'. The margin over the second-ranked "
            f"category was {margin:.4f}."
        )

        return KBSimilarityPrediction(
            suggested_ticket_type=top_type,
            confidence=confidence,
            similarity_scores=similarity_scores,
            top_reference_file=top_reference_file,
            explanation=explanation,
        )

    @staticmethod
    def _calculate_confidence(
        top_score: float,
        margin: float,
    ) -> float:
        """
        Convert cosine similarity and category separation into a
        conservative confidence score.

        This is not a calibrated probability.
        """
        normalized_similarity = max(min(top_score, 1.0), 0.0)
        normalized_margin = max(min(margin * 4, 1.0), 0.0)

        confidence = (
            normalized_similarity * 0.75
            + normalized_margin * 0.25
        )

        return round(confidence, 4)


def main() -> None:
    labeler = KBSimilarityLabeler()

    sample_tickets = [
        "My card keeps getting declined when I try to pay.",
        "Please stop my membership at the end of this month.",
        "Does this device work with Android phones?",
        "The item arrived damaged and I want my money returned.",
        "The application closes unexpectedly whenever I open it.",
        "I need help with my order.",
    ]

    for ticket in sample_tickets:
        result = labeler.predict(ticket)

        print("\n" + "=" * 100)
        print("Ticket:", ticket)
        print("Suggested Type:", result.suggested_ticket_type)
        print("Confidence:", result.confidence)
        print("Similarity Scores:", result.similarity_scores)
        print("Reference File:", result.top_reference_file)
        print("Explanation:", result.explanation)


if __name__ == "__main__":
    main()