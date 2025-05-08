# boutique_ai_project/Dockerfile
# Defines the container image for the Boutique AI backend engine.
# Includes Python, dependencies, and Playwright browsers.

# Use an official Python 3.11 slim base image for smaller size
FROM python:3.11-slim

# Set environment variables for Python and Playwright
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Set path for Playwright browsers within the container
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Set working directory inside the container
WORKDIR /app

# Install OS-level dependencies required by Playwright browsers
# This list is based on Playwright's recommendations for Debian-based images (like python:slim)
# Keeping it updated is important if Playwright requirements change.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Browsers dependencies:
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

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers with their dependencies
# This step downloads browser binaries and can take time / increase image size significantly.
# It's necessary if using the MultiModalPlaywrightAutomator.
RUN playwright install --with-deps

# Copy the entire application code into the container's working directory
# Ensure .dockerignore is used if needed to exclude files (like .env, venv, .git)
COPY . .

# Expose the port the application will listen on (should match LOCAL_SERVER_PORT in .env)
# Use ARG to allow overriding during build time if needed, but default to 8080
ARG APP_PORT=8080
ENV PORT=${APP_PORT}
EXPOSE ${PORT}

# Define the default command to run the application using Uvicorn via Python module execution
# This ensures the correct Python environment is used.
# Use the PORT environment variable set above.
# Add --workers based on production needs (e.g., 2 * CPU cores + 1)
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
# Example for production with 2 workers:
# CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]