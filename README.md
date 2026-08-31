# vectorize-for-ai

# need to add service account OAuth in GDrive into Shared memebers in google drive

## OpenSearch

Start container

```
docker run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<secret>" \
  -v opensearch-data:/usr/share/opensearch/data \
  opensearchproject/opensearch:latest
```

without user/password

```
podman run -d \
  --name opensearch \
  -p 9200:9200 -p 5601:5601 \
  -e discovery.type=single-node \
  -e plugins.security.disabled=true \
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
