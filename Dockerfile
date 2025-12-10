# Use Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget gnupg curl unzip git \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libxss1 \
    libasound2 libpangocairo-1.0-0 libpango-1.0-0 \
    libcairo2 libxshmfence1 libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Install Playwright + Chromium
RUN playwright install --with-deps chromium

# Copy code
COPY . /app
WORKDIR /app

# Expose HF Spaces port
ENV PORT=7860

# Start FastAPI
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
