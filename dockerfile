# boutique_ai_project/Dockerfile
# Defines the container image for the Boutique AI backend engine.
# Includes Python 3.11, dependencies via requirements.txt, and Playwright browsers.

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# Set environment variables for Python and Playwright
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Set path for Playwright browsers within the container
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Set working directory
WORKDIR /app

# Stage 2: Install OS dependencies
FROM base as os-deps
# Install OS-level dependencies required by Playwright browsers and potentially other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools if any C extensions need compiling (less likely with wheels)
    # build-essential \
    # Playwright browser dependencies (Debian/Ubuntu base):
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    # Fonts often needed for rendering pages correctly in headless mode:
    fonts-liberation \
    # Utilities needed for playwright install --with-deps:
    curl \
    unzip \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Install Python dependencies
FROM os-deps as python-deps
# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .
# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 4: Install Playwright browsers
FROM python-deps as playwright-install
# Install Playwright browsers with their dependencies using the installed playwright package
# This step downloads browser binaries and can take time / increase image size significantly.
RUN playwright install --with-deps

# Stage 5: Final application image
FROM base as final
# Copy necessary artifacts from previous stages
COPY --from=os-deps / /
COPY --from=python-deps /opt/venv /opt/venv # Assuming pip installs to a standard location or adjust if needed. If pip installs globally, this might not be needed. Let's assume global install for slim image.
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin
COPY --from=playwright-install ${PLAYWRIGHT_BROWSERS_PATH} ${PLAYWRIGHT_BROWSERS_PATH}

# Set working directory
WORKDIR /app

# Copy the entire application code into the container's working directory
COPY . .

# Expose the port the application will listen on (should match LOCAL_SERVER_PORT in .env)
ARG APP_PORT=8080
ENV PORT=${APP_PORT}
EXPOSE ${PORT}

# Define the default command to run the application using Uvicorn via Python module execution
# This ensures the correct Python environment is used.
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
# Example for production with 2 workers:
# CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]