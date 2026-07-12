from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ensemble_label_suggestions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reviewed"
    / "ticket_review_queue.csv"
)


SAMPLE_TARGETS = {
    "Technical issue": 50,
    "Product inquiry": 30,
    "Refund request": 30,
    "Cancellation request": 30,
    "Billing inquiry": 30,
}


def sample_group(
    df: pd.DataFrame,
    ticket_type: str,
    sample_size: int,
) -> pd.DataFrame:
    """Sample available records for one suggested ticket type."""
    group = df[
        df["final_suggested_ticket_type"] == ticket_type
    ]

    if group.empty:
        return group

    return group.sample(
        n=min(sample_size, len(group)),
        random_state=42,
    )


def create_review_queue() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ensemble output not found: {INPUT_PATH}. "
            "Run `python -m src.labeling.process_tickets` first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "ticket_id",
        "source_ticket_type",
        "ticket_text",
        "final_suggested_ticket_type",
        "ensemble_confidence",
        "decision_status",
        "review_priority",
        "original_label_mismatch",
        "decision_explanation",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    samples = []

    for ticket_type, sample_size in SAMPLE_TARGETS.items():
        sampled_group = sample_group(
            df=df,
            ticket_type=ticket_type,
            sample_size=sample_size,
        )

        samples.append(sampled_group)

    unresolved = df[
        df["final_suggested_ticket_type"].isna()
        | (
            df["decision_status"].isin(
                [
                    "HUMAN_REVIEW_REQUIRED",
                    "INSUFFICIENT_EVIDENCE",
                ]
            )
        )
    ]

    if not unresolved.empty:
        unresolved_sample = unresolved.sample(
            n=min(30, len(unresolved)),
            random_state=42,
        )

        samples.append(unresolved_sample)

    review_df = pd.concat(
        samples,
        ignore_index=True,
    )

    review_df = review_df.drop_duplicates(
        subset=["ticket_id"]
    )

    review_df["review_status"] = "PENDING"
    review_df["reviewed_ticket_type"] = ""
    review_df["review_action"] = ""
    review_df["review_notes"] = ""
    review_df["reviewed_by"] = ""
    review_df["reviewed_at"] = ""

    review_df = review_df.sort_values(
        by=[
            "review_priority",
            "ensemble_confidence",
        ],
        ascending=[True, False],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Review Queue Summary")
    print("=" * 70)
    print(f"Tickets added to queue: {len(review_df)}")

    print("\nSuggested ticket types:")
    print(
        review_df["final_suggested_ticket_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDecision statuses:")
    print(
        review_df["decision_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(f"\nReview queue saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_review_queue()