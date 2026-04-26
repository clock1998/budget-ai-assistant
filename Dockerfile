FROM python:3.12-slim

# System dependencies required by PyMuPDF and cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmupdf-dev \
        libssl-dev \
        libffi-dev \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source
# to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Gradio listens on 7860 by default
EXPOSE 7860

# GOOGLE_API_KEY must be supplied at runtime, e.g.:
#   docker run -e GOOGLE_API_KEY=<key> -p 7860:7860 budget-ai-assistant
ENV GOOGLE_API_KEY=""

# Run from /app so that src/ is on the path
CMD ["python", "src/ui.py"]
