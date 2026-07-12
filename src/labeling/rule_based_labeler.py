import re
from dataclasses import dataclass, asdict
import keyword
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
        """
        Match keywords and phrases using word boundaries.

        Multi-word phrases may contain up to two intermediate words.
        For example, 'card declined' can match:
        - card declined
        - card was declined
        - card keeps getting declined
        """
        keywords = self.taxonomy[ticket_type]["keywords"]
        matches = []

        for keyword in keywords:
            keyword_words = keyword.lower().split()

            if len(keyword_words) == 1:
                pattern = rf"(?<!\w){re.escape(keyword_words[0])}(?!\w)"

            else:
                escaped_words = [
                    re.escape(word)
                    for word in keyword_words
                ]

                separator = r"(?:\W+\w+){0,2}\W+"
                phrase_pattern = separator.join(escaped_words)

                pattern = rf"(?<!\w){phrase_pattern}(?!\w)"

            if re.search(pattern, text):
                matches.append(keyword)

        return matches

    def _calculate_scores(
        self,
        matched_keywords: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Assign weighted scores based on matched keyword strength.

        Multi-word phrases receive more weight because they usually
        represent stronger intent than generic single-word matches.
        """
        scores = {}

        for ticket_type in self.ticket_types:
            matches = matched_keywords[ticket_type]

            score = 0.0

            for keyword in matches:
                word_count = len(keyword.split())

                if word_count >= 3:
                    score += 3.0
                elif word_count == 2:
                    score += 2.0
                else:
                    score += 1.0

            scores[ticket_type] = round(score, 4)

        return scores

    @staticmethod
    def _calculate_confidence(
        category_scores: Dict[str, float]
    ) -> float:
        """
        Calculate confidence using the top score and separation
        from the second-ranked category.

        This is a heuristic confidence score, not a calibrated probability.
        """
        ranked_scores = sorted(
            category_scores.values(),
            reverse=True
        )

        if not ranked_scores or ranked_scores[0] == 0:
            return 0.0

        top_score = ranked_scores[0]
        second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0

        margin = max(top_score - second_score, 0.0)

        top_strength = min(top_score / 3.0, 1.0)
        margin_strength = min(margin / 2.0, 1.0)

        confidence = (
            top_strength * 0.65
            + margin_strength * 0.35
        )

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