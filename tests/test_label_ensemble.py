import pytest

from src.labeling.label_ensemble import (
    DecisionStatus,
    TicketLabelEnsemble,
)


@pytest.fixture(scope="module")
def ensemble():
    return TicketLabelEnsemble()


def test_agreeing_labelers_return_final_type(ensemble):
    result = ensemble.predict(
        ticket_text="My card payment was declined.",
        original_ticket_type="Billing inquiry",
    )

    assert result.final_suggested_ticket_type == "Billing inquiry"
    assert result.labelers_agree is True
    assert result.ensemble_confidence > 0


def test_original_mismatch_is_detected(ensemble):
    result = ensemble.predict(
        ticket_text="The device is broken and not working.",
        original_ticket_type="Refund request",
    )

    assert result.final_suggested_ticket_type == "Technical issue"
    assert result.original_label_mismatch is True


def test_vague_ticket_is_not_auto_accepted(ensemble):
    result = ensemble.predict(
        ticket_text="I need help with my order.",
        original_ticket_type="Billing inquiry",
    )

    assert result.decision_status != DecisionStatus.AUTO_ACCEPT.value


def test_empty_ticket_returns_insufficient_evidence(ensemble):
    result = ensemble.predict(
        ticket_text="",
        original_ticket_type="Technical issue",
    )

    assert result.final_suggested_ticket_type is None
    assert (
        result.decision_status
        == DecisionStatus.INSUFFICIENT_EVIDENCE.value
    )


def test_cancellation_mismatch_requires_review(ensemble):
    result = ensemble.predict(
        ticket_text="Please cancel my subscription.",
        original_ticket_type="Technical issue",
    )

    assert result.final_suggested_ticket_type == "Cancellation request"
    assert result.original_label_mismatch is True
    assert result.decision_status in {
        DecisionStatus.REVIEW_RECOMMENDED.value,
        DecisionStatus.HUMAN_REVIEW_REQUIRED.value,
    }


def test_prediction_can_convert_to_dict(ensemble):
    result = ensemble.predict(
        ticket_text="I want my money back.",
        original_ticket_type="Refund request",
    )

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert "final_suggested_ticket_type" in result_dict