---
doc_id: mobile-auth-policy-v3
title: Mobile Authorization Policy
doc_type: policy
department: compliance
language: en
access_level: internal
status: approved
valid_from: 2025-01-01
version: 3
---

# Mobile Authorization Policy

## Purpose

This policy governs how NovaBanque customers must authenticate high-risk operations
via the NovaBanque mobile application. It applies to all customer-facing transaction
flows and is reviewed annually by the Compliance team.

## Device Registration

Before a device can be used to authorise transactions, it must be registered through
the following process:

1. Log in with username and password on the target device.
2. Complete an SMS one-time password (OTP) challenge sent to the registered phone number.
3. Set a 6-digit PIN unique to this device (different from your card PIN).
4. Optionally enable biometric authentication (Face ID or fingerprint).

A maximum of **3 registered devices** are permitted per account at any time. Registering
a new device automatically de-registers the oldest device if the limit is reached.

## High-Risk Operations Requiring Step-Up Authentication

The following operations require step-up authentication (biometric or PIN confirmation)
regardless of whether the session is already active:

| Operation | Authentication Required |
|---|---|
| Transfer above €1,000 | Biometric or PIN + SMS OTP |
| Foreign transfer above €500 | Biometric + SMS OTP |
| Adding a new payee | Biometric or PIN |
| Changing account email or phone | Biometric + SMS OTP + 24 h delay |
| Closing the account | Biometric + SMS OTP + 48 h delay |
| Exporting full transaction history | PIN or Biometric |

## Lost or Stolen Device Procedure

If a customer loses access to a registered device, they must:

1. Contact NovaBanque support immediately via the web portal or phone (+48 800 123 456).
2. The support agent will initiate a **Device Lockout** on all registered devices,
   preventing any transaction authorisation for 24 hours.
3. The customer must complete an enhanced identity check (KYC video call or branch visit).
4. After verification, the customer may register a new device. All previous device
   registrations are permanently revoked.

During the Device Lockout period, read-only account access is available via the web
portal using username + password + email OTP only.

## Failed Authentication Attempts

- **3 consecutive failed PIN attempts**: device session is terminated; user must re-enter
  password.
- **5 consecutive failed PIN attempts**: device is de-registered and must go through the
  full registration process again.
- **10 failed login attempts in 24 hours**: account is temporarily suspended; customer must
  contact support to re-enable it.

## Session Management

- Mobile sessions expire after **15 minutes** of inactivity.
- Sessions are invalidated immediately on logout, password change, or device de-registration.
- A maximum of **1 active session per device** is enforced. Opening the app on a second
  browser or app instance on the same device terminates the first session.

## Policy Review

This policy is reviewed by the Compliance and Security teams every 12 months or
whenever a significant security incident occurs. The next scheduled review is
January 2026.
