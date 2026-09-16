# Dockerfile — SmartIDPhotoProcessor Production Container
# CPU-only inference with bundled local models
#
# Build:
#   docker build -t smartidphotoprocessor .
#
# Run:
#   docker run -p 8000:8000 smartidphotoprocessor
#
# Verify:
#   curl http://localhost:8000/api/v1/health
#   curl http://localhost:8000/api/v1/ready

# ---------------------------------------------------------------------------
# Stage 1: Build dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first (layer caching)
COPY requirements-pytorch-cpu.txt requirements-prod.txt ./
# Install PyTorch CPU-only first (avoid CUDA packages ~2GB), then remaining deps
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements-pytorch-cpu.txt --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install \
    -r requirements-prod.txt

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS production

# System runtime dependencies only
# libgl1 replaces libgl1-mesa-glx (deprecated in newer Debian)
# libglib2.0-0 required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Install the application package
COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-deps . \
    && find /app -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Copy model artifacts
COPY ai_models/ /app/ai_models/

# Environment configuration
ENV SMARTIDPHOTO_BASE_DIR=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the API port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Health check — process liveness only (cheap, no model loading)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Startup command — single worker, sync ML in async endpoint
CMD ["uvicorn", "smartidphotoprocessor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
