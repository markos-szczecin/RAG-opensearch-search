---
doc_id: compliance-notes-v4
title: Compliance Notes
doc_type: compliance
department: compliance
language: en
access_level: confidential
status: approved
valid_from: 2025-01-01
version: 4
---

# Compliance Notes

*For internal use by Compliance Officers and Senior Management only.*

## Regulatory Framework

NovaBanque operates under the following key regulations:

| Regulation | Scope | NovaBanque obligation |
|---|---|---|
| PSD2 (EU 2015/2366) | Payment services | Strong Customer Authentication (SCA), open banking APIs |
| GDPR (EU 2016/679) | Personal data | Lawful processing, data minimisation, right to erasure |
| AMLD5 (EU 2018/843) | Anti-money laundering | CDD, EDD for high-risk customers, STR filing |
| EBA Guidelines on PSD2 | SCA exemptions | Transaction Risk Analysis (TRA) thresholds |
| BFG Act (Poland) | Deposit protection | Contribution to deposit guarantee fund |

## KYC / CDD Requirements

### Standard Customer Due Diligence (CDD)

Required for all new customers:
- Full name, date of birth, nationality.
- Verified residential address.
- Source of funds declaration (self-certified for retail customers).
- Identity document verification (government-issued photo ID).

### Enhanced Due Diligence (EDD)

EDD applies to customers classified as high-risk, including:
- Politically Exposed Persons (PEPs) and their close associates.
- Customers from FATF high-risk jurisdictions.
- Customers whose transaction patterns trigger AML alerts.
- Business customers with complex ownership structures (> 3 layers).

EDD measures include:
- Senior management approval before account opening.
- Verification of source of wealth (not just source of funds).
- Ongoing monitoring at 6-month intervals (vs. 12-month for standard CDD).
- Annual revalidation of identity documents.

## Anti-Money Laundering (AML) Controls

### Transaction Monitoring Rules (examples — not exhaustive)

| Rule ID | Trigger | Action |
|---|---|---|
| AML-001 | Single transaction ≥ €10,000 | Auto-flag for AML review |
| AML-002 | Cumulative to same beneficiary ≥ €25,000 in 30 days | Auto-flag |
| AML-003 | Multiple small transactions adding to ≥ €10,000 in 24 h | Structuring alert |
| AML-004 | Transfer to FATF high-risk jurisdiction | EDD review required |
| AML-005 | Unusual geographic pattern (3+ countries in 24 h) | Manual review |

### Suspicious Transaction Reporting (STR)

If the Compliance team determines a transaction is suspicious after AML review,
an STR must be filed with the Polish Financial Intelligence Unit (GIIF) within
**2 business days** of the determination. Filing is confidential — the customer
must not be informed ("tipping off" prohibition under AMLD5 Article 39).

## GDPR Data Handling

### Retention Periods

| Data category | Retention period | Legal basis |
|---|---|---|
| Transaction records | 5 years from transaction date | AML/AML5D obligation |
| Customer identity documents | 5 years from account closure | AML5D obligation |
| Support call recordings | 12 months | Legitimate interest |
| Marketing consents | Until withdrawn + 1 year | GDPR Art. 6(1)(a) |
| Application logs | 90 days | Legitimate interest |

### Right to Erasure

Customers may request erasure of personal data. Erasure must be completed within
**30 days** unless a legal retention obligation applies (e.g. AML records). The
compliance team reviews each request to determine applicable exemptions before
any data is deleted.

## PSD2 Strong Customer Authentication (SCA)

SCA requires at least two of:
- **Knowledge**: password or PIN.
- **Possession**: registered device or OTP sent to registered phone.
- **Inherence**: biometric (fingerprint, face recognition).

### SCA Exemptions

The following are exempt from SCA under EBA Guidelines (subject to TRA thresholds):
- Contactless payments ≤ €50 (cumulative limit €150 or 5 consecutive transactions).
- Trusted beneficiary payments (pre-approved by customer).
- Recurring payments of identical amount to the same payee.
- Low-value remote transactions ≤ €30 (subject to TRA approval).

TRA exemptions are suspended if NovaBanque's fraud rate for that payment category
exceeds the threshold defined in EBA/GL/2019/01.

## Incident Reporting

Security and data incidents must be reported to the Compliance team within:
- **Data breach**: 4 hours (internal); 72 hours to UODO (Polish DPA) if high risk to individuals.
- **Operational incident affecting payments**: 4 hours (internal); next business day to KNF.
- **PSD2 major operational incident**: within 1 hour to NBP and KNF.
