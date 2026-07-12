# AI Ticket Type Correction and Routing Pipeline

## Overview

This project detects potentially mislabeled customer support tickets, suggests corrected ticket types, and routes uncertain cases to a human review workflow.

It extends the findings of the AI Support Copilot project, where manual investigation showed that several support tickets had labels that did not match their actual content.

## Business Problem

Incorrectly categorized support tickets can:

- Route customers to the wrong support team
- Increase resolution time
- Reduce first-contact resolution
- Distort support analytics
- Produce poor-quality machine learning models
- Increase manual reassignment work

## Proposed Solution

The system will:

1. Read existing support ticket records.
2. Analyze the ticket subject and description.
3. Suggest a revised ticket type.
4. Compare the suggested type with the original type.
5. Assign a confidence score.
6. Flag uncertain or conflicting records for human review.
7. Save approved labels as a cleaner training dataset.
8. Train an improved ticket-routing classifier.

## Target Ticket Types

- Billing inquiry
- Cancellation request
- Product inquiry
- Refund request
- Technical issue

## Planned Components

- Dataset and label quality audit
- Knowledge-base similarity scoring
- Rule-based classification signals
- Suggested ticket type generation
- Confidence scoring
- Label mismatch detection
- Human-in-the-loop Streamlit review interface
- MySQL audit trail
- Improved classifier trained on approved labels
- FastAPI inference service

## Architecture

```text
Original Ticket
      ↓
Text Cleaning
      ↓
Label Suggestion Engine
      ↓
Suggested Ticket Type + Confidence
      ↓
Compare with Original Ticket Type
      ↓
High Confidence Match ──→ Accept
      ↓
Mismatch or Uncertainty
      ↓
Human Review Queue
      ↓
Approved Revised Label
      ↓
Clean Training Dataset
      ↓
Improved Routing Model

## Project Status

Project setup and reusable component migration are in progress.

## Related Project

This project extends:

AI Support Copilot with RAG  https://github.com/usmandilshad-tech/ai-support-copilot-rag 

## Author

Muhammad Usman Dilshad

