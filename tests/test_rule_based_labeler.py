from src.labeling.rule_based_labeler import RuleBasedTicketLabeler


labeler = RuleBasedTicketLabeler()


def test_billing_prediction():
    result = labeler.predict(
        "My payment was declined and the card transaction failed."
    )

    assert result.suggested_ticket_type == "Billing inquiry"
    assert result.confidence > 0
    assert "payment" in result.matched_keywords["Billing inquiry"]


def test_cancellation_prediction():
    result = labeler.predict(
        "Please cancel my subscription."
    )

    assert result.suggested_ticket_type == "Cancellation request"


def test_product_inquiry_prediction():
    result = labeler.predict(
        "Is this product compatible with Windows?"
    )

    assert result.suggested_ticket_type == "Product inquiry"


def test_refund_prediction():
    result = labeler.predict(
        "I want a refund and my money back."
    )

    assert result.suggested_ticket_type == "Refund request"


def test_technical_prediction():
    result = labeler.predict(
        "The device is not working and makes a strange noise."
    )

    assert result.suggested_ticket_type == "Technical issue"


def test_empty_text_returns_no_prediction():
    result = labeler.predict("")

    assert result.suggested_ticket_type is None
    assert result.confidence == 0.0


def test_ambiguous_text_returns_no_prediction():
    result = labeler.predict(
        "I need help with my order."
    )

    assert result.suggested_ticket_type is None