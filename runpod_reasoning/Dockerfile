FROM python:3.11-slim

WORKDIR /app
COPY reason_service.py reasoning_schema.json system_prompt.md few_shot_examples.jsonl /app/
COPY eval /app/eval

RUN pip install --no-cache-dir fastapi uvicorn pydantic

ENV SAFELENS_REASONING_ARTIFACTS=/app
EXPOSE 8000
CMD ["uvicorn", "reason_service:app", "--host", "0.0.0.0", "--port", "8000"]
