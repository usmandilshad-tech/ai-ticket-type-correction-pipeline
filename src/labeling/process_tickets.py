from pathlib import Path

import pandas as pd

from src.labeling.label_ensemble import TicketLabelEnsemble


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
    / "ensemble_label_suggestions.csv"
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

    ensemble = TicketLabelEnsemble()

    predictions = []

    total = len(df)

    for index, row in df.iterrows():
        prediction = ensemble.predict(
            ticket_text=str(row["ticket_text"] or ""),
            original_ticket_type=row["ticket_type"],
        )

        predictions.append(prediction.to_dict())

        if (index + 1) % 250 == 0:
            print(
                f"Processed {index + 1:,} of {total:,} tickets...",
                flush=True,
            )

    prediction_df = pd.DataFrame(predictions)

    result_df = pd.concat(
        [
            df[[
                "ticket_id",
                "ticket_type",
                "ticket_text",
            ]].reset_index(drop=True),
            prediction_df.reset_index(drop=True),
        ],
        axis=1,
    )

    result_df = result_df.rename(
        columns={
            "ticket_type": "source_ticket_type",
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nEnsemble Processing Summary")
    print("=" * 70)
    print(f"Total tickets: {len(result_df):,}")

    print("\nDecision status:")
    print(
        result_df["decision_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nReview priority:")
    print(
        result_df["review_priority"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nOriginal label mismatches:",
        int(result_df["original_label_mismatch"].sum()),
    )

    print("\nFinal suggested ticket types:")
    print(
        result_df["final_suggested_ticket_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_tickets()