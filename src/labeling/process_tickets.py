from pathlib import Path

import pandas as pd

from src.labeling.kb_similarity_labeler import KBSimilarityLabeler
from src.labeling.rule_based_labeler import RuleBasedTicketLabeler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_support_tickets.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "combined_label_suggestions.csv"
)


def process_tickets() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "ticket_id",
        "ticket_type",
        "ticket_text",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    rule_labeler = RuleBasedTicketLabeler()
    kb_labeler = KBSimilarityLabeler()

    ticket_texts = df["ticket_text"].fillna("").astype(str)

    print("Generating rule-based suggestions...")
    rule_results = ticket_texts.apply(rule_labeler.predict)

    print("Generating knowledge-base similarity suggestions...")
    kb_results = ticket_texts.apply(kb_labeler.predict)

    df["suggested_ticket_type_rule"] = rule_results.apply(
        lambda result: result.suggested_ticket_type
    )
    df["rule_confidence"] = rule_results.apply(
        lambda result: result.confidence
    )
    df["rule_explanation"] = rule_results.apply(
        lambda result: result.explanation
    )

    df["suggested_ticket_type_kb"] = kb_results.apply(
        lambda result: result.suggested_ticket_type
    )
    df["kb_confidence"] = kb_results.apply(
        lambda result: result.confidence
    )
    df["kb_explanation"] = kb_results.apply(
        lambda result: result.explanation
    )

    df["labelers_agree"] = (
        df["suggested_ticket_type_rule"]
        == df["suggested_ticket_type_kb"]
    )

    df["rule_matches_original"] = (
        df["ticket_type"]
        == df["suggested_ticket_type_rule"]
    )

    df["kb_matches_original"] = (
        df["ticket_type"]
        == df["suggested_ticket_type_kb"]
    )

    df["review_required"] = (
        df["suggested_ticket_type_rule"].isna()
        | (~df["labelers_agree"])
        | (df["rule_confidence"] < 0.20)
        | (df["kb_confidence"] < 0.35)
        | (~df["rule_matches_original"])
        | (~df["kb_matches_original"])
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "ticket_id",
        "ticket_type",
        "suggested_ticket_type_rule",
        "rule_confidence",
        "suggested_ticket_type_kb",
        "kb_confidence",
        "labelers_agree",
        "rule_matches_original",
        "kb_matches_original",
        "review_required",
        "rule_explanation",
        "kb_explanation",
        "ticket_text",
    ]

    df[output_columns].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nProcessing Summary")
    print("=" * 60)
    print(f"Total tickets: {len(df)}")
    print(
        "Rule suggestions generated:",
        df["suggested_ticket_type_rule"].notna().sum(),
    )
    print(
        "KB suggestions generated:",
        df["suggested_ticket_type_kb"].notna().sum(),
    )
    print(
        "Labeler agreements:",
        df["labelers_agree"].sum(),
    )
    print(
        "Rule matches original:",
        df["rule_matches_original"].sum(),
    )
    print(
        "KB matches original:",
        df["kb_matches_original"].sum(),
    )
    print(
        "Tickets requiring review:",
        df["review_required"].sum(),
    )
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_tickets()