FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CRAWL4AI_CACHE_DIR=/app/.cache/crawl4ai
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    libu2f-udev \
    libvulkan1 \
    libxml2-dev \
    libxslt1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create cache directories
RUN mkdir -p /app/.cache/crawl4ai /app/.cache/huggingface

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# crawl4ai[all] includes sentence-transformers and other ML dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install crawl4ai with all extras for embedding support
RUN pip install --no-cache-dir "crawl4ai[all]==0.7.8"

# Run crawl4ai setup to install browsers and download models
RUN crawl4ai-setup

# Copy application code
COPY . .

# Expose port (Zeabur will use PORT env variable)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health', timeout=5)" || exit 1

# Start the application
CMD ["python", "main.py"]
