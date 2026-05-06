---
doc_id: transfer-procedures-v2
title: Transfer Procedures
doc_type: procedure
department: operations
language: en
access_level: internal
status: approved
valid_from: 2025-03-01
version: 2
---

# Transfer Procedures

## Domestic SEPA Transfers

### Standard SEPA Credit Transfer (SCT)

1. Navigate to **Payments → Send Money → SEPA Transfer**.
2. Enter the recipient's IBAN and BIC (BIC auto-filled if IBAN is recognised).
3. Enter amount and reference (max 140 characters).
4. Review and confirm with biometric or PIN if amount > €1,000.
5. Execution: funds are credited within **1 business day**.

### SEPA Instant Credit Transfer (SCT Inst)

SCT Inst is available 24/7 for amounts up to **€100,000**. Funds are credited within
10 seconds. Not all receiving banks support SCT Inst; the app shows availability
before confirmation.

## Foreign (SWIFT) Transfers

### Eligibility

- Available to Standard, Premium, Business Basic, and Business Pro accounts.
- Student accounts cannot initiate foreign transfers.
- Joint accounts require both account holders to confirm transfers above €5,000.

### Process

1. Navigate to **Payments → Send Money → International Transfer**.
2. Enter recipient name, IBAN or account number, SWIFT/BIC, and bank address.
3. Select currency. Exchange rate is locked for 30 seconds after preview.
4. For transfers above €10,000, the system triggers an AML screening check.
   This may add 1–3 business days to the processing time.
5. Submit. A fee of €15 plus correspondent bank charges applies.

### AML Screening for Large Transfers

Transfers above **€10,000** in a single transaction, or **€25,000** cumulative in
a 30-day rolling window to the same beneficiary, are automatically flagged for
AML review by the Operations team.

During review, the transfer is held in a pending state. The customer is notified
by email within 1 business day of the outcome:
- **Approved**: funds are released immediately.
- **Additional information required**: customer receives a secure message listing
  required documentation (e.g. invoice, contract, purpose of transfer).
- **Declined**: funds are returned to the sender's account. A compliance officer
  may contact the customer for further information.

## Recurring Transfers (Standing Orders)

1. Navigate to **Payments → Recurring Payments → New Standing Order**.
2. Configure recipient, amount, frequency (weekly, monthly, quarterly), start date,
   and optional end date or fixed number of executions.
3. Standing orders execute at **08:00 CET** on the scheduled date.
4. If the account has insufficient funds on the execution date, the payment is
   retried at 14:00 CET and 20:00 CET. After 3 failures, the standing order is
   suspended and the customer is notified.

## Transfer Limits

Refer to the [Account Limits](account_limits.csv) document for per-account-type limits.
Transfer amounts exceeding account limits are rejected at the submission stage with a
clear error message indicating the applicable limit.

## Cancellation Policy

- **SEPA SCT**: cancellable up to the processing cut-off time (18:00 CET on execution day).
- **SCT Inst**: cannot be cancelled once accepted.
- **SWIFT**: cancellable within 2 hours of submission if not yet processed by the
  correspondent bank. A recall fee of €25 applies.
