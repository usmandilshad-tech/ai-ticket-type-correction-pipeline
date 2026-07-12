import pytest

from src.labeling.kb_similarity_labeler import KBSimilarityLabeler


@pytest.fixture(scope="module")
def labeler():
    return KBSimilarityLabeler()


@pytest.mark.parametrize(
    "ticket_text, expected_type",
    [
        (
            "My card payment was declined.",
            "Billing inquiry",
        ),
        (
            "Please cancel and close my subscription.",
            "Cancellation request",
        ),
        (
            "Is this product compatible with Android?",
            "Product inquiry",
        ),
        (
            "I want my money back and need a refund.",
            "Refund request",
        ),
        (
            "The application crashes and does not work.",
            "Technical issue",
        ),
    ],
)
def test_expected_predictions(labeler, ticket_text, expected_type):
    result = labeler.predict(ticket_text)

    assert result.suggested_ticket_type == expected_type
    assert result.confidence > 0
    assert expected_type in result.similarity_scores


def test_empty_text(labeler):
    result = labeler.predict("")

    assert result.suggested_ticket_type is None
    assert result.confidence == 0.0


def test_similarity_scores_include_all_categories(labeler):
    result = labeler.predict(
        "The product is broken and displays an error."
    )

    assert len(result.similarity_scores) == 5