# Multi-stage build for lean production image
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies.
# Le fichier est coupé au marqueur "=== DEV" : l'image ne sert que
# l'application Streamlit. Embarquer les outils de test et de notebook
# l'alourdirait de plusieurs centaines de mégaoctets et y ajouterait un
# serveur Jupyter inutile.
COPY requirements.txt .
RUN sed '/^# === DEV/q' requirements.txt > requirements-runtime.txt \
    && pip install --user --no-cache-dir -r requirements-runtime.txt


# Final stage - minimal production image
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    LOG_FILE=/app/logs/app.log

# Create logs directory
RUN mkdir -p /app/logs

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
