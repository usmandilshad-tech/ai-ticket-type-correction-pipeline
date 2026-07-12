from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reviewed"
    / "ticket_review_queue.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reviewed"
    / "approved_ticket_labels.csv"
)


def export_reviewed_labels() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Review queue not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    completed = df[
        df["review_status"]
        .fillna("")
        .str.upper()
        .eq("COMPLETED")
    ].copy()

    completed = completed[
        completed["reviewed_ticket_type"]
        .fillna("")
        .str.strip()
        .ne("")
    ]

    output_columns = [
        "ticket_id",
        "ticket_text",
        "source_ticket_type",
        "final_suggested_ticket_type",
        "reviewed_ticket_type",
        "review_action",
        "review_notes",
        "reviewed_by",
        "reviewed_at",
    ]

    completed[output_columns].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Completed reviews exported: {len(completed)}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_reviewed_labels()