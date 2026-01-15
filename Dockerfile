# ============================================================================
# PRODUCTION-READY DOCKERFILE - Crawl4AI v5.0.0
# Multi-Stage Build | Security Hardened | Production Best Practices
# ============================================================================

# ============================================================================
# STAGE 1: BUILDER - Compile dependencies
# ============================================================================
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /builder

COPY requirements.txt .

# Pre-compile wheels for faster installation
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /builder/wheels -r requirements.txt

# ============================================================================
# STAGE 2: RUNTIME - Minimal production image
# ============================================================================
FROM python:3.11-slim

LABEL maintainer="your-team@example.com" \
      version="5.0.0" \
      description="Crawl4AI - Two-Phase Hybrid Crawler"

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CRAWL4AI_HEADLESS=true \
    PLAYWRIGHT_BROWSERS_PATH=/app/.browsers \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    PORT=8080 \
    LOG_FILE=/app/logs/crawl4ai.log \
    LOG_LEVEL=INFO \
    DEEPSEEK_API_KEY="" \
    OPENROUTER_API_KEY="" \
    CRAWL_API_KEY="" \
    REQUIRE_API_KEY="false" \
    CORS_ORIGINS="*"

# ============================================================================
# SYSTEM DEPENDENCIES (For Playwright/Chromium)
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    fonts-liberation \
    fonts-dejavu-core \
    libasound2 \
    libpulse0 \
    libopus0 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libgtk-3-common \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxss1 \
    libxtst6 \
    libnspr4 \
    libnss3 \
    libu2f-udev \
    libvulkan1 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# CREATE APPLICATION DIRECTORIES
# ============================================================================
RUN mkdir -p /app /app/logs /app/.cache /app/.browsers && \
    chmod 755 /app

WORKDIR /app

# ============================================================================
# INSTALL PYTHON DEPENDENCIES
# ============================================================================
COPY --from=builder /builder/wheels /tmp/wheels
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt && \
    rm -rf /tmp/wheels

# ============================================================================
# CRAWL4AI INSTALLATION & SETUP
# ============================================================================
RUN pip install --no-cache-dir 'crawl4ai[all]>=0.3.0' && \
    crawl4ai-setup 2>&1 | tail -10

# ============================================================================
# PLAYWRIGHT BROWSER INSTALLATION
# ============================================================================
RUN playwright install chromium --with-deps 2>&1 | tail -10

# ============================================================================
# COPY APPLICATION CODE
# ============================================================================
COPY main.py /app/main.py

# Verify syntax
RUN python -m py_compile /app/main.py

# ============================================================================
# CLEANUP
# ============================================================================
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/* && \
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete && \
    find /app -type f -name "*.pyo" -delete && \
    pip cache purge

# ============================================================================
# SECURITY: NON-ROOT USER
# ============================================================================
RUN groupadd -r crawl4ai && useradd -r -g crawl4ai -u 1001 crawl4ai && \
    chown -R crawl4ai:crawl4ai /app && \
    chmod -R 750 /app && \
    chmod 777 /app/logs /app/.cache /app/.browsers

USER crawl4ai

# ============================================================================
# HEALTH CHECK
# ============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# ============================================================================
# EXPOSE PORT
# ============================================================================
EXPOSE ${PORT}

# ============================================================================
# VOLUMES
# ============================================================================
VOLUME ["/app/logs", "/app/.cache", "/app/.browsers"]

# ============================================================================
# ENTRY POINT
# ============================================================================
CMD ["python", "main.py"]
