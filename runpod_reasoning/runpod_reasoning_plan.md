# RunPod Reasoning Service Plan

## Contract

`POST /reason` accepts detections plus selected frame, company, and zone context.
It returns the typed `ReasoningRecord` validated by both Pydantic and
`reasoning_schema.json`. The service advises the LangGraph workflow and never
executes an HSE action.

## Candidate Models And GPU Sizing

| Candidate | License checkpoint | Starting GPU | Purpose |
|---|---|---:|---|
| Qwen2.5-VL 7B Instruct | Apache-2.0; reverify exact checkpoint | 24 GB L4/A10 | Default accuracy/latency baseline |
| Qwen2.5-VL 3B Instruct | Apache-2.0; reverify exact checkpoint | 16 GB T4/A4000 | Lower-cost latency baseline |
| Molmo 7B-D | Apache-2.0; reverify exact checkpoint | 24 GB L4/A10 | Grounding and pointing comparison |
| InternVL 2.5 8B | checkpoint-specific; verify before use | 24-48 GB L4/A6000 | Spatial-reasoning comparison |

Start with a 24 GB L4 or A10 pod. Use quantization only after schema reliability,
critical false-negative rate, and grounding quality are compared with the
unquantized baseline.

## Build And Launch

```bash
docker build -t safelens-reasoning:0.2 .
docker run --rm -p 8000:8000 safelens-reasoning:0.2
curl http://localhost:8000/health
```

On RunPod, keep the endpoint private, mount model cache storage, inject model
credentials as secrets, and expose it only through the signed Cloudflare
internal route. Configure the worker with `SAFELENS_REASONING_URL`; never accept
an origin URL from a browser request.

## VLM Adapter

Load the selected model once at startup. Build the request from:

- `system_prompt.md`
- `few_shot_examples.jsonl`
- retrieved company/site rules
- minimum-necessary visual evidence
- detector and zone context

Parse the generated JSON into `ReasoningRecord`. Reject malformed output,
recompute score/band/approval deterministically, and route any outage or invalid
response to mandatory human review.

## Evaluation

Run every entry in `eval/context_pairs.json` plus all few-shot examples. Measure:

- schema-valid output rate
- score, band, and approval consistency
- safe/unsafe pair ordering
- latent versus active classification
- unsupported scene or standards claims
- critical false-negative rate
- p50/p95 latency and peak GPU memory

Acceptance requires 100% schema validity, zero approval-threshold inconsistencies,
and human QHSE sign-off on every critical pair. No model is deployed
automatically.

## Security And Operations

- Require signed service-to-service requests and replay protection.
- Limit body size, media type, request duration, and generated tokens.
- Use short-lived evidence references and redact secrets/private content.
- Record prompt/model/config versions without logging full customer evidence.
- Separate training jobs from live reasoning capacity or enforce queue limits.
