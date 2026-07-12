# Ticket Type Labeling Guidelines

## Purpose

These guidelines define how customer support tickets should be categorized.

They are used by:

- Rule-based labeling logic
- Knowledge-base similarity labeling
- Human reviewers
- Model training and evaluation

## General Principle

Assign the ticket type based on the customer's primary requested action or main problem.

When a ticket contains multiple issues, use the issue that requires the first or most important operational action.

## Ticket Types

### Billing inquiry

Use when the customer is asking about:

- Charges
- Invoices
- Payment failures
- Duplicate payments
- Billing details
- Payment methods

Do not use when the customer explicitly requests a refund.

### Cancellation request

Use when the customer wants to:

- Cancel an order
- Cancel a subscription
- Stop a service
- Close an account

If the customer also requests money back, review whether `Refund request` is the primary action.

### Product inquiry

Use when the customer asks about:

- Features
- Specifications
- Compatibility
- Availability
- Pricing
- Setup or usage instructions

Do not use if the customer reports a confirmed malfunction.

### Refund request

Use when the customer explicitly asks for:

- A refund
- Money back
- Reimbursement
- Refund status

A refund request may arise from billing, cancellation, delivery, or product problems, but the requested outcome is repayment.

### Technical issue

Use when the customer reports:

- A system error
- Login or password failure
- Product malfunction
- Hardware problem
- Software bug
- Intermittent behavior
- Application or device failure

## Multi-Issue Tickets

When more than one category appears:

1. Identify the customer's requested outcome.
2. Identify the operational team that should act first.
3. Prefer explicit intent over background context.
4. Flag ambiguous cases for human review.

Examples:

| Ticket | Recommended Type |
|---|---|
| "I was charged twice and need a refund" | Refund request |
| "My payment failed" | Billing inquiry |
| "Cancel my subscription and refund the last payment" | Refund request |
| "Is this compatible with Windows?" | Product inquiry |
| "The product makes strange noises" | Technical issue |