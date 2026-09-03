FROM registry.access.redhat.com/ubi10/python-312-minimal:latest

USER root

# Create yum cache dir (required in rootless Podman builds) then install deps
RUN mkdir -p /var/cache/yum/metadata && \
    microdnf install -y shadow-utils libxcb mesa-libGL ghostscript openssl openssl-devel gcc-c++ make && \
    microdnf clean all && \
    useradd -m -u 1001 appuser

WORKDIR /app

# Copy source and install all dependencies via PEP 517 build
COPY --chown=appuser:appuser pyproject.toml README.md app.py mcp_app.py ./
COPY --chown=appuser:appuser src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

USER 1001
