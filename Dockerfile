FROM python:3.12-slim AS base

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- dependency layer (cached unless pyproject.toml changes) ----
FROM base AS deps
COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

# ---- runtime image ----
FROM deps AS runtime
COPY app/ ./app/
COPY scripts/ ./scripts/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
