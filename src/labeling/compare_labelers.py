from src.labeling.kb_similarity_labeler import KBSimilarityLabeler
from src.labeling.rule_based_labeler import RuleBasedTicketLabeler


def main() -> None:
    rule_labeler = RuleBasedTicketLabeler()
    kb_labeler = KBSimilarityLabeler()

    sample_tickets = [
        "My card keeps getting declined when I try to pay.",
        "Please cancel my subscription immediately.",
        "Will this product work with Windows 11?",
        "The product arrived broken and I want a refund.",
        "My device is making unusual noises and stops working.",
        "I need help with my order.",
    ]

    for ticket in sample_tickets:
        rule_result = rule_labeler.predict(ticket)
        kb_result = kb_labeler.predict(ticket)

        agreement = (
            rule_result.suggested_ticket_type
            == kb_result.suggested_ticket_type
        )

        print("\n" + "=" * 100)
        print("Ticket:", ticket)
        print("-" * 100)
        print(
            "Rule Suggestion:",
            rule_result.suggested_ticket_type,
            "| Confidence:",
            rule_result.confidence,
        )
        print(
            "KB Suggestion:",
            kb_result.suggested_ticket_type,
            "| Confidence:",
            kb_result.confidence,
        )
        print("Labelers Agree:", agreement)
        print("Rule Explanation:", rule_result.explanation)
        print("KB Explanation:", kb_result.explanation)


if __name__ == "__main__":
    main()