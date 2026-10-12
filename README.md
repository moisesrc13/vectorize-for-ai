# vectorize-for-ai

# need to add service account OAuth in GDrive into Shared memebers in google drive

## OpenSearch

Start container

```
podman run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<secret>" \
  -v opensearch-data:/usr/share/opensearch/data \
  opensearchproject/opensearch:latest
```

## Open WebUI

podman run -d -p 3000:8080 -v open-webui-dev:/app/backend/data --name open-webui-dev ghcr.io/open-webui/open-webui:dev

[local link](http://0.0.0.0:3000)

### with OpenSearch

podman run -d -p 3000:8080 \
  -e VECTOR_DB=opensearch \
  -e OPENSEARCH_URI="http://docker.internal" \
  -e OPENSEARCH_USERNAME="admin" \
  -e OPENSEARCH_PASSWORD="<pwd>" \
  -v open-webui-dev:/app/backend/data \
  --name open-webui-dev \
  --restart always \
  ghcr.io/open-webui/open-webui:main

## MCP Server

`python mcp_app.py`


connecting from OpenWEB ui, point to `http://host.docker.internal:8443/mcp`

## On ARM Cpu

podman build --platform linux/arm64 -t vectorize-for-ai:rpi -f Dockerfile_rpi .
# or with docker buildx:
docker buildx build --platform linux/arm64 -t vectorize-for-ai:rpi -f Dockerfile_rpi --load .

# To run the API service
podman run -d \
  --name vectorize-api \
  -p 8010:8010 \
  --env-file .env \
  vectorize-for-ai:rpi

# Or to run the MCP server
podman run -d \
  --name vectorize-mcp \
  -p 8443:8443 \
  --env-file .env \
  vectorize-for-ai:rpi python mcp_app.py

## Ollama setup
Open the systemd override file for Ollama:

sudo systemctl edit ollama

Add these lines between the comments:

[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"

Reload systemd and restart the Ollama service:

sudo systemctl daemon-reload
sudo systemctl restart ollama

Verify it is now bound to all interfaces:

ss -tulpn | grep 11434
# You should see: 0.0.0.0:11434 or *:11434 instead of 127.0.0.1:11434
