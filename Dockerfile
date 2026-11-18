# Single image for the API, the registry bootstrap, the MLflow UI and the demo.
# Git stays in the image: each MLflow run logs the real commit (traceability requirement).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m app \
    && mkdir -p /app /mlflow \
    && chown app:app /app /mlflow \
    && git config --system --add safe.directory /app

WORKDIR /app
USER app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PATH="/app/.venv/bin:$PATH"

# Dependencies first: this layer is only rebuilt when uv.lock changes.
COPY --chown=app:app pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY --chown=app:app . .
RUN uv sync --locked --no-dev \
    && mkdir -p mlartifacts data/processed data/monitoring

ENV UV_NO_SYNC=1
EXPOSE 8000
CMD ["uvicorn", "pga.api:app", "--host", "0.0.0.0", "--port", "8000"]
