# Use an official Python base image
FROM python:3.11-slim

# Set environment variables to prevent buffering issues with logs
ENV PYTHONUNBUFFERED=1 \
    # Playwright specific env vars (optional, might help in some environments)
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright 

# Set working directory
WORKDIR /app

# Install OS dependencies needed by Playwright browsers
# This list might need adjustment based on the base image and specific needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Common dependencies for Chromium, Firefox, WebKit
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    # Fonts often needed
    fonts-liberation \
    # Utility
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
# IMPORTANT: This downloads browser binaries into the image, making it larger.
RUN playwright install --with-deps

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
# Use ARG to make it easily configurable if needed, but default to 8080
ARG PORT=8080
ENV PORT=${PORT}
EXPOSE ${PORT}

# Define the command to run the application
# Use the python -m uvicorn method for robustness
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
# For production, consider more workers:
# CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"] 