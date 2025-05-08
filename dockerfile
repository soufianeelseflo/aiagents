# /Dockerfile:
# --------------------------------------------------------------------------------
# Base image: Choose a specific Python version for reproducibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Set the working directory in the container
WORKDIR ${APP_HOME}

# Install system dependencies
# - Basic tools like git (if needed for any pip installs from VCS)
# - Dependencies for Playwright (Playwright will install its own browsers, but needs OS libs)
#   The `playwright install --with-deps` command later handles browser-specific OS dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Add any other essential OS packages here if needed by your application
    # For example, if any of your Python packages have C extensions that need specific build tools
    # or libraries not covered by Playwright's deps.
    # For now, keeping it minimal as Playwright handles its own.
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (if you were using it, but requirements.txt is provided)
# RUN pip install poetry

# Copy only essential files for installing dependencies first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --break-system-packages for newer pip versions with externally managed environments
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --break-system-packages

# Install Playwright browsers and their OS dependencies
# IMPORTANT: Choose the browser you need (chromium, firefox, webkit) or install all
# Using chromium as an example. --with-deps is crucial for installing OS libraries.
RUN playwright install --with-deps chromium
# If you need multiple:
# RUN playwright install --with-deps chromium firefox webkit

# Copy the rest of the application code
COPY . .

# Create a non-root user and group
# RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup -d ${APP_HOME} -s /sbin/nologin appuser
# Change ownership of the app directory
# RUN chown -R appuser:appgroup ${APP_HOME}
# Switch to the non-root user
# USER appuser
# Note: Running as non-root is best practice, but for simpler Coolify setups,
# you might run as root if the VPS is dedicated and permissions are managed.
# If running as non-root, ensure directories like /app/logs (if used) are writable by appuser.

# Expose the port the app runs on (ensure this matches LOCAL_SERVER_PORT in your .env/config.py)
# Defaulting to 8080 as per original README guidance for Coolify.
EXPOSE 8080

# Command to run the application using Uvicorn
# This CMD will be overridden if you set a "Start Command" in Coolify,
# but it's good practice to have a default.
# It references LOCAL_SERVER_PORT from config.py which should load from .env
# The host 0.0.0.0 makes it accessible from outside the container.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
# If config.LOCAL_SERVER_PORT is reliably set and you want to use it directly:
# CMD sh -c "uvicorn server:app --host 0.0.0.0 --port ${LOCAL_SERVER_PORT:-8080}"
# This ^ requires LOCAL_SERVER_PORT to be an environment variable available at runtime.
# The current server.py logic uses config.LOCAL_SERVER_PORT for the uvicorn.run call
# when __name__ == "__main__", but for Docker CMD, explicitly stating the port is safer
# unless you pass LOCAL_SERVER_PORT as an ENV variable to the container.
# The Python code itself (config.py) will load LOCAL_SERVER_PORT from .env for the app.
# --------------------------------------------------------------------------------