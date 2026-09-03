FROM registry.access.redhat.com/ubi10/python-312-minimal:latest

USER root

# Create yum cache dir (required in rootless Podman builds) then install deps
RUN mkdir -p /var/cache/yum/metadata && \
    microdnf install -y shadow-utils libxcb mesa-libGL ghostscript openssl openssl-devel gcc-c++ make && \
    microdnf clean all && \
    useradd -m -u 1001 appuser

WORKDIR /app

# Install Poetry and project dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copy source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py mcp_app.py ./

USER 1001
