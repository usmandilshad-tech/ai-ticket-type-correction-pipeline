from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from src.labeling.label_taxonomy import TICKET_TAXONOMY, TICKET_TYPES


@dataclass
class RuleBasedPrediction:
    suggested_ticket_type: Optional[str]
    confidence: float
    matched_keywords: Dict[str, List[str]]
    category_scores: Dict[str, float]
    explanation: str

    def to_dict(self) -> Dict:
        return asdict(self)


class RuleBasedTicketLabeler:
    """
    Suggest a ticket type using transparent keyword-based rules.

    This is intended as one signal in a larger human-in-the-loop
    labeling pipeline. It should not silently overwrite existing labels.
    """

    def __init__(self):
        self.ticket_types = TICKET_TYPES
        self.taxonomy = TICKET_TAXONOMY

    @staticmethod
    def _normalise_text(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _find_keyword_matches(
        self,
        text: str,
        ticket_type: str
    ) -> List[str]:
        keywords = self.taxonomy[ticket_type]["keywords"]

        matches = [
            keyword
            for keyword in keywords
            if keyword.lower() in text
        ]

        return matches

    def _calculate_scores(
        self,
        matched_keywords: Dict[str, List[str]]
    ) -> Dict[str, float]:
        scores = {}

        for ticket_type in self.ticket_types:
            category_keywords = self.taxonomy[ticket_type]["keywords"]
            matches = matched_keywords[ticket_type]

            if not category_keywords:
                scores[ticket_type] = 0.0
                continue

            raw_score = len(matches) / len(category_keywords)
            scores[ticket_type] = round(raw_score, 4)

        return scores

    @staticmethod
    def _calculate_confidence(
        category_scores: Dict[str, float]
    ) -> float:
        sorted_scores = sorted(
            category_scores.values(),
            reverse=True
        )

        if not sorted_scores or sorted_scores[0] == 0:
            return 0.0

        top_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        margin = top_score - second_score

        confidence = (top_score * 0.7) + (margin * 0.3)

        return round(min(confidence, 1.0), 4)

    def predict(self, ticket_text: str) -> RuleBasedPrediction:
        if not ticket_text or not ticket_text.strip():
            return RuleBasedPrediction(
                suggested_ticket_type=None,
                confidence=0.0,
                matched_keywords={
                    ticket_type: []
                    for ticket_type in self.ticket_types
                },
                category_scores={
                    ticket_type: 0.0
                    for ticket_type in self.ticket_types
                },
                explanation="No ticket text was provided."
            )

        normalised_text = self._normalise_text(ticket_text)

        matched_keywords = {
            ticket_type: self._find_keyword_matches(
                normalised_text,
                ticket_type
            )
            for ticket_type in self.ticket_types
        }

        category_scores = self._calculate_scores(matched_keywords)

        suggested_ticket_type = max(
            category_scores,
            key=category_scores.get
        )

        if category_scores[suggested_ticket_type] == 0:
            suggested_ticket_type = None

        confidence = self._calculate_confidence(category_scores)

        if suggested_ticket_type is None:
            explanation = (
                "No category-specific keywords were found. "
                "The ticket should be sent for human review."
            )
        else:
            top_matches = matched_keywords[suggested_ticket_type]

            explanation = (
                f"Suggested '{suggested_ticket_type}' because the ticket "
                f"matched the following keywords: {', '.join(top_matches)}."
            )

        return RuleBasedPrediction(
            suggested_ticket_type=suggested_ticket_type,
            confidence=confidence,
            matched_keywords=matched_keywords,
            category_scores=category_scores,
            explanation=explanation
        )


def main() -> None:
    labeler = RuleBasedTicketLabeler()

    sample_tickets = [
        "My payment was declined and I cannot complete the transaction.",
        "Please cancel my subscription immediately.",
        "Is this product compatible with Windows 11?",
        "I want a refund and my money back.",
        "The device is not working and makes a strange noise.",
        "I need help with my order."
    ]

    for ticket in sample_tickets:
        result = labeler.predict(ticket)

        print("\n" + "=" * 100)
        print("Ticket:", ticket)
        print("Suggested Type:", result.suggested_ticket_type)
        print("Confidence:", result.confidence)
        print("Matched Keywords:", result.matched_keywords)
        print("Category Scores:", result.category_scores)
        print("Explanation:", result.explanation)


if __name__ == "__main__":
    main()