from typing import Dict, List


TICKET_TYPES = [
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue",
]


TICKET_TAXONOMY: Dict[str, Dict[str, List[str] | str]] = {
    "Billing inquiry": {
        "definition": (
            "Questions or problems related to charges, invoices, payment methods, "
            "billing details, failed payments, or duplicate charges."
        ),
        "include_when": [
            "Customer asks about an invoice or billing statement",
            "Customer reports a failed or declined payment",
            "Customer reports an incorrect or duplicate charge",
            "Customer asks about payment methods or billing details",
            "Customer questions a subscription charge",
        ],
        "exclude_when": [
            "Customer explicitly asks for money to be returned",
            "Customer asks to cancel an order or subscription",
            "Customer reports a product malfunction",
        ],
        "keywords": [
            "billing issue",
            "billing question",
            "billing statement",
            "invoice",
            "missing invoice",
            "incorrect invoice",
            "charged twice",
            "charged more than once",
            "duplicate charge",
            "incorrect charge",
            "wrong charge",
            "unexpected charge",
            "unrecognized charge",
            "subscription charge",
            "payment declined",
            "payment was declined",
            "payment keeps getting declined",
            "card declined",
            "card was declined",
            "card keeps getting declined",
            "payment failed",
            "payment did not go through",
            "payment won't go through",
            "transaction failed",
            "transaction was declined",
            "unable to pay",
            "cannot make payment",
            "billing address",
            "payment method",
            "credit card",
            "debit card",
        ],
    },

    "Cancellation request": {
        "definition": (
            "Requests to cancel an order, subscription, service, booking, "
            "or account before or after fulfillment."
        ),
        "include_when": [
            "Customer asks to cancel a subscription",
            "Customer asks to cancel an order",
            "Customer asks to stop a service",
            "Customer asks to close an account",
        ],
        "exclude_when": [
            "Customer only asks for a refund",
            "Customer asks why a payment failed",
            "Customer reports a technical problem without requesting cancellation",
        ],
        "keywords": [
            "cancel",
            "cancellation",
            "stop subscription",
            "end subscription",
            "close account",
            "terminate",
            "unsubscribe",
        ],
    },

    "Product inquiry": {
        "definition": (
            "Questions about product features, compatibility, availability, "
            "specifications, setup, pricing, or general usage before a confirmed fault."
        ),
        "include_when": [
            "Customer asks whether a product is compatible",
            "Customer asks about product specifications",
            "Customer asks about features or availability",
            "Customer asks for setup or usage information",
            "Customer asks about pricing before purchase",
        ],
        "exclude_when": [
            "Customer reports that the product is broken or malfunctioning",
            "Customer requests a refund",
            "Customer reports a payment problem",
        ],
        "keywords": [
            "features",
            "specifications",
            "compatible",
            "compatibility",
            "availability",
            "price",
            "pricing",
            "how to use",
            "setup",
            "product information",
            "work with",
            "works with",
            "support windows",
            "compatible with",
            "can i use",
            "does this work",
        ],
    },

    "Refund request": {
        "definition": (
            "Requests for money to be returned due to dissatisfaction, accidental "
            "purchase, non-delivery, duplicate charge, cancellation, or product issues."
        ),
        "include_when": [
            "Customer explicitly asks for a refund",
            "Customer asks for money back",
            "Customer requests reimbursement",
            "Customer wants a refund after cancellation",
            "Customer requests a refund for a defective or undelivered product",
        ],
        "exclude_when": [
            "Customer only asks about an invoice",
            "Customer asks to cancel but does not request money back",
            "Customer reports a technical problem without asking for a refund",
        ],
        "keywords": [
            "refund",
            "money back",
            "reimbursement",
            "return my money",
            "refund request",
            "refund status",
        ],
    },

    "Technical issue": {
        "definition": (
            "Problems where a product, application, account, device, or service is "
            "not working correctly, including bugs, errors, login failures, hardware "
            "faults, and intermittent problems."
        ),
        "include_when": [
            "Customer cannot log in or reset a password",
            "Customer reports an error message or software bug",
            "Customer reports that a product is not working",
            "Customer reports a hardware fault",
            "Customer reports intermittent or unexpected behavior",
        ],
        "exclude_when": [
            "Customer only asks about product features",
            "Customer only asks about billing",
            "Customer only requests cancellation or refund",
        ],
        "keywords": [
            "not working",
            "error",
            "bug",
            "login",
            "password",
            "hardware",
            "broken",
            "malfunction",
            "intermittent",
            "strange noise",
            "crash",
            "failed to load",
            "stops working",
            "not turning on",
            "not charging",
            "unusual noise",
            "making noise",
            "does not respond",
        ],
    },
}


def get_ticket_types() -> List[str]:
    """Return the supported ticket types."""
    return TICKET_TYPES.copy()


def get_taxonomy_for_type(ticket_type: str) -> Dict:
    """Return taxonomy details for a ticket type."""
    if ticket_type not in TICKET_TAXONOMY:
        raise ValueError(f"Unsupported ticket type: {ticket_type}")

    return TICKET_TAXONOMY[ticket_type]


def validate_ticket_type(ticket_type: str) -> bool:
    """Check whether a ticket type is supported."""
    return ticket_type in TICKET_TYPES


def print_taxonomy() -> None:
    """Print taxonomy definitions for review."""
    for ticket_type, rules in TICKET_TAXONOMY.items():
        print("\n" + "=" * 100)
        print(ticket_type)
        print("=" * 100)
        print("Definition:", rules["definition"])

        print("\nInclude when:")
        for item in rules["include_when"]:
            print(f"- {item}")

        print("\nExclude when:")
        for item in rules["exclude_when"]:
            print(f"- {item}")

        print("\nKeywords:")
        print(", ".join(rules["keywords"]))


if __name__ == "__main__":
    print_taxonomy()