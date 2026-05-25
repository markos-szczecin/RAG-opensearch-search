---
doc_id: api-authentication-guide-for-developers-v1
title: API Authentication Guide for Developers
doc_type: developer
department: engineering
language: en
access_level: internal
status: approved
valid_from: 2025-01-01
version: 1
---

# API Authentication Guide for Developers

## Overview

NovaBanque's API authentication system is built on industry-standard OAuth 2.0 and mutual TLS certificate validation. This guide provides developers with comprehensive instructions for implementing secure authentication across all NovaBanque API endpoints.

All API requests must include proper authentication credentials. Requests without valid authentication will be rejected with a 401 Unauthorized response.

## Authentication Methods

### OAuth 2.0 Client Credentials Flow

The **Client Credentials flow** is the primary authentication method for server-to-server communication. This flow is ideal for backend services that need to access NovaBanque APIs without user interaction.

**Implementation Steps:**

1. Register your application in the Developer Portal
2. Generate a unique **Client ID** and **Client Secret**
3. Exchange credentials for an access token via the `/oauth/token` endpoint
4. Include the access token in the `Authorization` header of API requests

**Example Request:**
```
POST /oauth/token HTTP/1.1
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET
```

Access tokens expire after **3600 seconds** (1 hour). Implement token refresh logic in your application to request new tokens before expiration.

### Mutual TLS (mTLS) Authentication

For high-security integrations, mutual TLS authentication provides certificate-based authentication without requiring bearer tokens. Both client and server validate each other's certificates.

**Setup Requirements:**

- Generate a **certificate signing request (CSR)** for your application
- Submit the CSR to NovaBanque's certificate authority
- Receive a signed certificate valid for **2 years**
- Configure your HTTP client to use the certificate for all API requests

Certificate renewal must occur before expiration. We recommend initiating renewal **90 days** before the expiry date.

## API Request Headers

All API requests must include the following headers:

| Header | Value | Required |
|--------|-------|----------|
| `Authorization` | `Bearer {access_token}` | Yes (OAuth 2.0) |
| `Content-Type` | `application/json` | Yes |
| `X-API-Version` | `2025-01` | Yes |
| `X-Request-ID` | UUID (unique per request) | Yes |

The `X-Request-ID` header improves debugging and transaction tracking. Generate a new UUID for each request.

## Rate Limiting

API requests are rate-limited to **1000 requests per minute** per authenticated application. Rate limit information is included in response headers:

- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

When the rate limit is exceeded, the API returns a **429 Too Many Requests** response.

## Security Best Practices

- **Never commit credentials** to version control. Use environment variables or secure configuration management systems.
- **Rotate secrets regularly** through the Developer Portal.
- **Use HTTPS exclusively** for all API communications.
- **Implement timeout logic** with a minimum of 30 seconds for long-running operations.
- **Log authentication failures** for security monitoring and incident response.
- **Validate SSL certificates** on the server side to prevent man-in-the-middle attacks.

## Troubleshooting

If you encounter authentication errors, verify:

1. Access token has not expired
2. Client credentials are correct and active
3. Request headers include all required fields
4. Your IP address is whitelisted (if applicable)
5. Certificate dates are current (for mTLS)

Contact the NovaBanque Developer Support team for additional assistance.