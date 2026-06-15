# Cloudflare Routing And Security Plan

## Objectives

Cloudflare is the public policy boundary. It authenticates users/services,
derives tenant context, rate-limits expensive operations, routes to private
origins, and emits audit telemetry without exposing RunPod pod URLs.

## Route Map

| Public path | Private destination | Policy |
|---|---|---|
| `/detect`, `/ws/vision` | SafeLens worker | normal vision limits |
| `/build/*` | SafeLens worker | authenticated Build Mode |
| `/agentic/*` | SafeLens worker | authenticated, tenant-bound |
| `/internal/reasoning/*` | RunPod reasoning origin | service-token only |
| `/internal/training/*` | RunPod job/status service | admin/service only |

The browser never receives a raw worker token, RunPod API key, pod hostname, or
direct model-weights URL.

## Authentication

For user requests:

1. Validate issuer, audience, signature, expiry, and not-before on the JWT.
2. Derive `owner_id` and allowed `org_id` from trusted claims or a trusted
   membership lookup.
3. Ignore client-supplied owner/organization fields where they disagree.
4. Add a short-lived signed internal identity header for the worker.

For service requests:

- use a rotated service token or asymmetric request signature
- include timestamp, nonce, method, path, and body digest
- reject replayed or stale requests
- scope tokens by route and environment

## Tenant Routing

Tenant context is carried in signed internal headers such as:

- `X-SafeLens-Owner-Id`
- `X-SafeLens-Org-Id`
- `X-SafeLens-Request-Id`
- `X-SafeLens-Service-Signature`

The worker treats these as trusted only when Cloudflare's signature validates.
The future organization RLS policy must use a real membership table, not an
arbitrary `org_id` claim from the client.

## Build Mode Resolution

The application resolves Build Mode in this order:

1. Fetch reviewed Build Mode configuration.
2. Call authenticated Cloudflare `/build/*`.
3. Use the existing local/mock fallback only for an explicitly allowed
   development or degraded mode.

Agentic reasoning does not replace this path.

## Reasoning Forwarding

The worker submits a minimal payload to a private Cloudflare internal route.
Cloudflare attaches the RunPod credential server-side and forwards to the active
reasoning origin. Responses are size-limited and schema-validated again by the
worker.

Frame references should be short-lived signed objects. Avoid forwarding full
video, unrelated frames, user tokens, or customer identifiers.

## Rate Limits

Suggested starting limits per user and organization:

- `/detect`: high sustained limit based on current product behavior
- `/build/session/frame`: bounded by existing session/frame controls
- `/agentic/reason`: low burst and low sustained limit
- draft routes: moderate interactive limit
- approval route: strict replay/idempotency controls
- internal training: administrator-only concurrency limit

Apply cost ceilings and timeouts to reasoning. A timeout becomes a logged
degraded result, not an implicit safe result.

## Logging

Log:

- request ID and route
- owner/org IDs
- model/config versions
- latency and status
- reasoning score/band
- approval requirement and decision reference

Do not log raw credentials, full image payloads, embeddings, private document
content, or unnecessary personal data. High and critical reasoning calls should
emit a security/audit event.

## Origin Protection

- restrict worker and RunPod ingress to Cloudflare/service identities
- use TLS end to end
- store secrets in Cloudflare bindings/secrets, never frontend code
- rotate service credentials
- apply response/body size limits
- disable directory and debug exposure
- use environment-specific origins and keys

## Failure Behavior

- invalid JWT/signature: `401`
- valid identity without permission: `403`
- tenant mismatch: `403` and audit event
- reasoning rate limit: `429` with retry metadata
- RunPod unavailable: `503`, logged fallback/review required
- invalid reasoning schema: `502` or `503`, never pass malformed output onward

## Deployment Sequence

1. Add staging origins and secrets.
2. Implement JWT and service-signature verification.
3. Route health and low-risk draft endpoints.
4. Add private reasoning forwarding.
5. Exercise tenant isolation and replay tests.
6. Enable structured high/critical audit logs.
7. Gradually expose authorized production workflows.
