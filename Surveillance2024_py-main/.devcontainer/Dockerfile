FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Install Poetry
ENV POETRY_VERSION=1.1.13
RUN curl -sSL https://install.python-poetry.org | python -

# Ensure that the Poetry bin directory is in the PATH
ENV PATH="${PATH}:/root/.local/bin"

# Copy your pyproject.toml and poetry.lock (if present) to the container
COPY pyproject.toml poetry.lock* /app/

# Set the working directory to /app
WORKDIR /app

# Install the project dependencies
RUN poetry install --no-root --no-dev

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Your additional Dockerfile commands...
