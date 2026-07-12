from src.labeling.label_taxonomy import (
    TICKET_TYPES,
    TICKET_TAXONOMY,
    get_taxonomy_for_type,
    validate_ticket_type,
)


def test_expected_ticket_types_exist():
    expected_types = {
        "Billing inquiry",
        "Cancellation request",
        "Product inquiry",
        "Refund request",
        "Technical issue",
    }

    assert set(TICKET_TYPES) == expected_types


def test_every_ticket_type_has_required_fields():
    required_fields = {
        "definition",
        "include_when",
        "exclude_when",
        "keywords",
    }

    for ticket_type, rules in TICKET_TAXONOMY.items():
        assert required_fields.issubset(rules.keys())
        assert rules["definition"]
        assert rules["include_when"]
        assert rules["exclude_when"]
        assert rules["keywords"]


def test_valid_ticket_type():
    assert validate_ticket_type("Technical issue") is True


def test_invalid_ticket_type():
    assert validate_ticket_type("Sales inquiry") is False


def test_get_taxonomy_for_type():
    taxonomy = get_taxonomy_for_type("Refund request")

    assert "refund" in taxonomy["keywords"]


def test_unsupported_type_raises_error():
    try:
        get_taxonomy_for_type("Unknown type")
        assert False, "Expected ValueError"
    except ValueError:
        assert True