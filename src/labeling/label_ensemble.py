from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional

from src.labeling.kb_similarity_labeler import (
    KBSimilarityLabeler,
    KBSimilarityPrediction,
)
from src.labeling.rule_based_labeler import (
    RuleBasedPrediction,
    RuleBasedTicketLabeler,
)


class DecisionStatus(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    REVIEW_RECOMMENDED = "REVIEW_RECOMMENDED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReviewPriority(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class EnsemblePrediction:
    original_ticket_type: Optional[str]
    rule_suggested_type: Optional[str]
    kb_suggested_type: Optional[str]
    final_suggested_ticket_type: Optional[str]
    rule_confidence: float
    kb_confidence: float
    ensemble_confidence: float
    labelers_agree: bool
    original_label_mismatch: bool
    decision_status: str
    review_priority: str
    decision_explanation: str
    rule_explanation: str
    kb_explanation: str

    def to_dict(self) -> Dict:
        return asdict(self)


class TicketLabelEnsemble:
    """
    Combine rule-based and semantic ticket type suggestions.

    The ensemble does not overwrite the original ticket label.
    It returns a recommendation and review decision.
    """

    def __init__(
        self,
        rule_labeler: Optional[RuleBasedTicketLabeler] = None,
        kb_labeler: Optional[KBSimilarityLabeler] = None,
    ):
        self.rule_labeler = rule_labeler or RuleBasedTicketLabeler()
        self.kb_labeler = kb_labeler or KBSimilarityLabeler()

    def predict(
        self,
        ticket_text: str,
        original_ticket_type: Optional[str] = None,
    ) -> EnsemblePrediction:
        rule_result = self.rule_labeler.predict(ticket_text)
        kb_result = self.kb_labeler.predict(ticket_text)

        return self._combine_predictions(
            original_ticket_type=original_ticket_type,
            rule_result=rule_result,
            kb_result=kb_result,
        )

    def _combine_predictions(
        self,
        original_ticket_type: Optional[str],
        rule_result: RuleBasedPrediction,
        kb_result: KBSimilarityPrediction,
    ) -> EnsemblePrediction:
        rule_type = rule_result.suggested_ticket_type
        kb_type = kb_result.suggested_ticket_type

        labelers_agree = (
            rule_type is not None
            and kb_type is not None
            and rule_type == kb_type
        )

        if rule_type is None and kb_result.confidence < 0.25:
            final_type = None
            ensemble_confidence = 0.0
            decision_status = DecisionStatus.INSUFFICIENT_EVIDENCE
            review_priority = ReviewPriority.MEDIUM
            explanation = (
                "The rule-based labeler found no category-specific evidence, "
                "and the semantic similarity confidence was too low to support "
                "a reliable recommendation."
            )

        elif labelers_agree:
            final_type = rule_type

            ensemble_confidence = round(
                min(
                    1.0,
                    (
                        rule_result.confidence * 0.45
                        + kb_result.confidence * 0.45
                        + 0.10
                    ),
                ),
                4,
            )

            if (
                rule_result.confidence >= 0.45
                and kb_result.confidence >= 0.40
            ):
                decision_status = DecisionStatus.AUTO_ACCEPT
                review_priority = ReviewPriority.NONE
                explanation = (
                    f"Both labelers independently suggested '{final_type}' "
                    "with sufficiently strong evidence."
                )
            else:
                decision_status = DecisionStatus.REVIEW_RECOMMENDED
                review_priority = ReviewPriority.LOW
                explanation = (
                    f"Both labelers suggested '{final_type}', but one or both "
                    "confidence scores were moderate."
                )

        elif rule_type is None and kb_result.confidence >= 0.45:
            final_type = kb_type
            ensemble_confidence = round(
                kb_result.confidence * 0.80,
                4,
            )
            decision_status = DecisionStatus.REVIEW_RECOMMENDED
            review_priority = ReviewPriority.MEDIUM
            explanation = (
                f"The semantic labeler suggested '{kb_type}' with meaningful "
                "confidence, but the rule-based labeler found no explicit evidence."
            )

        elif (
            rule_type is not None
            and kb_type is not None
            and rule_type != kb_type
        ):
            if (
                rule_result.confidence >= 0.55
                and rule_result.confidence
                >= kb_result.confidence + 0.15
            ):
                final_type = rule_type
                ensemble_confidence = round(
                    rule_result.confidence * 0.75,
                    4,
                )
                decision_status = DecisionStatus.HUMAN_REVIEW_REQUIRED
                review_priority = ReviewPriority.HIGH
                explanation = (
                    f"The labelers disagreed. The rule-based labeler strongly "
                    f"favored '{rule_type}', while the semantic labeler suggested "
                    f"'{kb_type}'. Human review is required."
                )

            elif (
                kb_result.confidence >= 0.55
                and kb_result.confidence
                >= rule_result.confidence + 0.15
            ):
                final_type = kb_type
                ensemble_confidence = round(
                    kb_result.confidence * 0.75,
                    4,
                )
                decision_status = DecisionStatus.HUMAN_REVIEW_REQUIRED
                review_priority = ReviewPriority.HIGH
                explanation = (
                    f"The labelers disagreed. The semantic labeler strongly "
                    f"favored '{kb_type}', while the rule-based labeler suggested "
                    f"'{rule_type}'. Human review is required."
                )

            else:
                final_type = None
                ensemble_confidence = 0.0
                decision_status = DecisionStatus.HUMAN_REVIEW_REQUIRED
                review_priority = ReviewPriority.HIGH
                explanation = (
                    f"The labelers disagreed between '{rule_type}' and "
                    f"'{kb_type}', and neither signal was strong enough to "
                    "recommend one safely."
                )

        else:
            final_type = kb_type or rule_type
            available_confidence = max(
                rule_result.confidence,
                kb_result.confidence,
            )

            ensemble_confidence = round(
                available_confidence * 0.65,
                4,
            )
            decision_status = DecisionStatus.REVIEW_RECOMMENDED
            review_priority = ReviewPriority.MEDIUM
            explanation = (
                "Only one labeler produced a usable suggestion. "
                "Human confirmation is recommended."
            )

        original_label_mismatch = (
            original_ticket_type is not None
            and final_type is not None
            and original_ticket_type != final_type
        )

        if (
            original_label_mismatch
            and decision_status == DecisionStatus.AUTO_ACCEPT
        ):
            decision_status = DecisionStatus.REVIEW_RECOMMENDED
            review_priority = ReviewPriority.MEDIUM
            explanation += (
                " The recommendation differs from the original ticket type, "
                "so review is recommended before updating the stored label."
            )

        return EnsemblePrediction(
            original_ticket_type=original_ticket_type,
            rule_suggested_type=rule_type,
            kb_suggested_type=kb_type,
            final_suggested_ticket_type=final_type,
            rule_confidence=rule_result.confidence,
            kb_confidence=kb_result.confidence,
            ensemble_confidence=ensemble_confidence,
            labelers_agree=labelers_agree,
            original_label_mismatch=original_label_mismatch,
            decision_status=decision_status.value,
            review_priority=review_priority.value,
            decision_explanation=explanation,
            rule_explanation=rule_result.explanation,
            kb_explanation=kb_result.explanation,
        )


def main() -> None:
    ensemble = TicketLabelEnsemble()

    sample_tickets = [
        {
            "original": "Billing inquiry",
            "text": "My card was declined when I tried to pay.",
        },
        {
            "original": "Refund request",
            "text": "My device makes strange noises and stops working.",
        },
        {
            "original": "Product inquiry",
            "text": "Will this device work with Windows 11?",
        },
        {
            "original": "Cancellation request",
            "text": "I need help with my order.",
        },
        {
            "original": "Technical issue",
            "text": "Please cancel my subscription immediately.",
        },
    ]

    for sample in sample_tickets:
        result = ensemble.predict(
            ticket_text=sample["text"],
            original_ticket_type=sample["original"],
        )

        print("\n" + "=" * 100)
        print("Ticket:", sample["text"])
        print("Original Type:", result.original_ticket_type)
        print("Rule Suggestion:", result.rule_suggested_type)
        print("KB Suggestion:", result.kb_suggested_type)
        print("Final Suggestion:", result.final_suggested_ticket_type)
        print("Ensemble Confidence:", result.ensemble_confidence)
        print("Decision Status:", result.decision_status)
        print("Review Priority:", result.review_priority)
        print("Mismatch:", result.original_label_mismatch)
        print("Explanation:", result.decision_explanation)


if __name__ == "__main__":
    main()