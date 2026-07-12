from pathlib import Path

import pandas as pd

from src.labeling.rule_based_labeler import RuleBasedTicketLabeler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_support_tickets.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "rule_based_label_suggestions.csv"


def process_tickets() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "ticket_id",
        "ticket_type",
        "ticket_text"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    labeler = RuleBasedTicketLabeler()

    results = df["ticket_text"].fillna("").apply(labeler.predict)

    df["suggested_ticket_type_rule"] = results.apply(
        lambda result: result.suggested_ticket_type
    )

    df["rule_confidence"] = results.apply(
        lambda result: result.confidence
    )

    df["rule_explanation"] = results.apply(
        lambda result: result.explanation
    )

    df["rule_label_matches_original"] = (
        df["ticket_type"] == df["suggested_ticket_type_rule"]
    )

    df["rule_review_required"] = (
        df["suggested_ticket_type_rule"].isna()
        | (df["rule_confidence"] < 0.25)
        | (~df["rule_label_matches_original"])
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "ticket_id",
        "ticket_type",
        "suggested_ticket_type_rule",
        "rule_confidence",
        "rule_label_matches_original",
        "rule_review_required",
        "rule_explanation",
        "ticket_text"
    ]

    df[output_columns].to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Processed tickets: {len(df)}")
    print(
        "Suggested labels generated:",
        df["suggested_ticket_type_rule"].notna().sum()
    )
    print(
        "Labels matching original:",
        df["rule_label_matches_original"].sum()
    )
    print(
        "Tickets requiring review:",
        df["rule_review_required"].sum()
    )
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_tickets()