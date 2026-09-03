FROM registry.redhat.io/ubi9/python-311-minimal:latest

# Create non-root user
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Install build tools needed by some Python packages
USER root
RUN microdnf install -y libxcb mesa-libGL ghostscript openssl openssl-devel gcc-c++ make && microdnf clean all

# Install Poetry and project dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Copy source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py mcp_app.py ./

USER 1001
