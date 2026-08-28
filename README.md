# vectorize-for-ai

# need to add service account OAuth in GDrive into Shared memebers in google drive

## OpenSearch

Start container

```
docker run -d \
  --name opensearch-node \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<secret>" \
  -v opensearch-data:/usr/share/opensearch/data \
  opensearchproject/opensearch:latest
```
