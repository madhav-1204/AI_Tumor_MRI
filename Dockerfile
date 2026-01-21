# Base image
FROM python:3.10-slim

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# System dependencies
# - libgl1, libglib2.0-0 → OpenCV
# - fonts-dejavu-core → matplotlib + PDF fonts
# - build-essential → native wheels safety
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache optimization)
COPY requirements.txt .

# Upgrade pip & install Python deps
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit runs here
EXPOSE 8501

# Streamlit environment (Docker-friendly)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_PORT=8501

# Start the app
CMD ["streamlit", "run", "app.py"]
