from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.labeling.label_taxonomy import TICKET_TYPES

REVIEW_QUEUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reviewed"
    / "ticket_review_queue.csv"
)


st.set_page_config(
    page_title="Ticket Label Review",
    page_icon="🏷️",
    layout="wide",
)


def load_review_queue() -> pd.DataFrame:
    if not REVIEW_QUEUE_PATH.exists():
        raise FileNotFoundError(
            f"Review queue not found: {REVIEW_QUEUE_PATH}. "
            "Run `python -m src.data.create_review_queue` first."
        )

    df = pd.read_csv(REVIEW_QUEUE_PATH)

    text_columns = [
        "review_status",
        "reviewed_ticket_type",
        "review_action",
        "review_notes",
        "reviewed_by",
        "reviewed_at",
    ]

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype(str)

    return df


def save_review_queue(df: pd.DataFrame) -> None:
    df.to_csv(REVIEW_QUEUE_PATH, index=False)


def get_pending_indices(df: pd.DataFrame) -> list[int]:
    return df.index[
        df["review_status"].str.upper() != "COMPLETED"
    ].tolist()


def clean_display_value(value) -> str:
    if pd.isna(value):
        return "Not available"

    value = str(value).strip()
    return value if value else "Not available"


if "review_df" not in st.session_state:
    try:
        st.session_state.review_df = load_review_queue()
    except Exception as exc:
        st.error(str(exc))
        st.stop()


if "current_position" not in st.session_state:
    st.session_state.current_position = 0


df = st.session_state.review_df
pending_indices = get_pending_indices(df)

st.title("Human-in-the-Loop Ticket Label Review")
st.caption(
    "Review AI-generated ticket type corrections before updating labels."
)

total_count = len(df)
completed_count = int(
    (df["review_status"].str.upper() == "COMPLETED").sum()
)
pending_count = total_count - completed_count

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Total Review Tickets", total_count)

with metric_col2:
    st.metric("Completed", completed_count)

with metric_col3:
    st.metric("Pending", pending_count)


if not pending_indices:
    st.success("All tickets in the review queue have been completed.")
    st.stop()


if st.session_state.current_position >= len(pending_indices):
    st.session_state.current_position = 0


current_index = pending_indices[st.session_state.current_position]
ticket = df.loc[current_index]


with st.sidebar:
    st.header("Review Controls")

    reviewer_name = st.text_input(
        "Reviewer name",
        value="Muhammad Usman Dilshad",
    )

    status_filter = st.selectbox(
        "View records",
        [
            "Pending only",
            "All records",
        ],
    )

    st.progress(
        completed_count / total_count
        if total_count > 0
        else 0
    )

    st.write(
        f"Reviewing pending ticket "
        f"{st.session_state.current_position + 1} "
        f"of {len(pending_indices)}"
    )


st.subheader(f"Ticket ID: {clean_display_value(ticket['ticket_id'])}")

meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

with meta_col1:
    st.metric(
        "Original Type",
        clean_display_value(ticket["source_ticket_type"]),
    )

with meta_col2:
    st.metric(
        "Final AI Suggestion",
        clean_display_value(
            ticket["final_suggested_ticket_type"]
        ),
    )

with meta_col3:
    st.metric(
        "Confidence",
        round(
            float(ticket["ensemble_confidence"]),
            3,
        ),
    )

with meta_col4:
    st.metric(
        "Review Priority",
        clean_display_value(ticket["review_priority"]),
    )


left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Ticket Text")
    st.text_area(
        "Customer message",
        value=clean_display_value(ticket["ticket_text"]),
        height=260,
        disabled=True,
    )

with right_col:
    st.markdown("### Label Comparison")

    comparison_df = pd.DataFrame(
        {
            "Source": [
                "Original label",
                "Rule-based suggestion",
                "KB similarity suggestion",
                "Ensemble suggestion",
            ],
            "Ticket Type": [
                clean_display_value(
                    ticket["source_ticket_type"]
                ),
                clean_display_value(
                    ticket["rule_suggested_type"]
                ),
                clean_display_value(
                    ticket["kb_suggested_type"]
                ),
                clean_display_value(
                    ticket["final_suggested_ticket_type"]
                ),
            ],
        }
    )

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True,
    )

    st.write(
        "**Decision status:**",
        clean_display_value(ticket["decision_status"]),
    )

    st.write(
        "**Original mismatch:**",
        bool(ticket["original_label_mismatch"]),
    )


with st.expander("Decision explanation"):
    st.write(
        clean_display_value(
            ticket["decision_explanation"]
        )
    )

with st.expander("Rule-based explanation"):
    st.write(
        clean_display_value(
            ticket["rule_explanation"]
        )
    )

with st.expander("KB similarity explanation"):
    st.write(
        clean_display_value(
            ticket["kb_explanation"]
        )
    )


st.markdown("---")
st.subheader("Reviewer Decision")

suggested_value = ticket["final_suggested_ticket_type"]

if pd.isna(suggested_value) or not str(suggested_value).strip():
    default_type = clean_display_value(
        ticket["source_ticket_type"]
    )
else:
    default_type = str(suggested_value)


review_action = st.radio(
    "Choose review action",
    [
        "Approve AI suggestion",
        "Keep original label",
        "Override with another label",
    ],
)


if review_action == "Approve AI suggestion":
    reviewed_ticket_type = default_type

elif review_action == "Keep original label":
    reviewed_ticket_type = str(
        ticket["source_ticket_type"]
    )

else:
    default_index = (
        TICKET_TYPES.index(default_type)
        if default_type in TICKET_TYPES
        else 0
    )

    reviewed_ticket_type = st.selectbox(
        "Select corrected ticket type",
        TICKET_TYPES,
        index=default_index,
    )


review_notes = st.text_area(
    "Review notes",
    placeholder=(
        "Optional: explain why the label was approved, "
        "rejected, or overridden."
    ),
)


button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    save_button = st.button(
        "Save Review",
        type="primary",
        width="stretch",
    )

with button_col2:
    previous_button = st.button(
        "Previous",
        width="stretch",
    )

with button_col3:
    skip_button = st.button(
        "Skip",
        width="stretch",
    )


if save_button:
    if (
        review_action == "Approve AI suggestion"
        and (
            pd.isna(ticket["final_suggested_ticket_type"])
            or not str(
                ticket["final_suggested_ticket_type"]
            ).strip()
        )
    ):
        st.error(
            "There is no AI suggestion to approve. "
            "Keep the original label or choose an override."
        )

    elif not reviewer_name.strip():
        st.error("Enter a reviewer name before saving.")

    else:
        df.at[current_index, "review_status"] = "COMPLETED"
        df.at[current_index, "reviewed_ticket_type"] = (
            reviewed_ticket_type
        )
        df.at[current_index, "review_action"] = (
            review_action
        )
        df.at[current_index, "review_notes"] = (
            review_notes.strip()
        )
        df.at[current_index, "reviewed_by"] = (
            reviewer_name.strip()
        )
        df.at[current_index, "reviewed_at"] = (
            datetime.now().isoformat(timespec="seconds")
        )

        save_review_queue(df)

        st.session_state.review_df = df

        st.success(
            f"Review saved as '{reviewed_ticket_type}'."
        )

        if st.session_state.current_position >= len(
            get_pending_indices(df)
        ):
            st.session_state.current_position = 0

        st.rerun()


if previous_button:
    if st.session_state.current_position > 0:
        st.session_state.current_position -= 1

    st.rerun()


if skip_button:
    if (
        st.session_state.current_position
        < len(pending_indices) - 1
    ):
        st.session_state.current_position += 1
    else:
        st.session_state.current_position = 0

    st.rerun()


st.markdown("---")
st.subheader("Review Queue Preview")

if status_filter == "Pending only":
    preview_df = df[
        df["review_status"].str.upper() != "COMPLETED"
    ]
else:
    preview_df = df

preview_columns = [
    "ticket_id",
    "source_ticket_type",
    "final_suggested_ticket_type",
    "ensemble_confidence",
    "decision_status",
    "review_priority",
    "review_status",
    "reviewed_ticket_type",
]

st.dataframe(
    preview_df[preview_columns],
    width="stretch",
    hide_index=True,
)