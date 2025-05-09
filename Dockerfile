# /Dockerfile: (Production Ready - Holistic Review)
# --------------------------------------------------------------------------------
# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the application home directory
ENV APP_HOME /app
WORKDIR ${APP_HOME}

# Install system dependencies
# - curl is needed for health checks or other simple downloads.
# - Other dependencies are installed by 'playwright install --with-deps'
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    # Add any other essential OS packages that your Python packages might need for compilation
    # For example, build-essential, libpq-dev if using psycopg2 directly, etc.
    # For now, keeping it minimal as Playwright handles its own browser dependencies.
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip and Poetry (optional, if you migrate to Poetry later)
# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir poetry

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Using --break-system-packages for compatibility with newer pip in system Python images
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --break-system-packages

# Install Playwright browsers and their operating system dependencies.
# This is a critical step. Using chromium as the default.
# The '--with-deps' flag is essential for installing necessary OS libraries.
RUN playwright install --with-deps chromium
# If you need other browsers, uncomment and add them:
# RUN playwright install --with-deps firefox webkit

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (must match Uvicorn command and config)
EXPOSE 8080

# Default command to run the application using Uvicorn.
# This will be used if no "Start Command" is specified in Coolify.
# Ensures the app is accessible from outside the container on the exposed port.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
# --------------------------------------------------------------------------------