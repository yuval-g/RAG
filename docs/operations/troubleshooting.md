# RAG Engine Troubleshooting & FAQ

This document provides solutions to common issues and frequently asked questions about the RAG Engine.

## Table of Contents

1. [Quick Troubleshooting](#quick-troubleshooting)
2. [Installation Issues](#installation-issues)
3. [Configuration Problems](#configuration-problems)
4. [API and Authentication](#api-and-authentication)
5. [Document Indexing Issues](#document-indexing-issues)
6. [Query Processing Problems](#query-processing-problems)
7. [Performance Issues](#performance-issues)
8. [Memory and Resource Problems](#memory-and-resource-problems)
9. [Deployment Issues](#deployment-issues)
10. [Monitoring and Logging](#monitoring-and-logging)
11. [Frequently Asked Questions](#frequently-asked-questions)

## Quick Troubleshooting

### Health Check Commands

```bash
# Check system health
curl http://localhost:8089/health

# Check API status
curl http://localhost:8000/api/v1/health

# Check system info
curl http://localhost:8089/status

# Check logs
docker-compose logs rag-engine
# or
kubectl logs deployment/rag-engine -n rag-engine
```

### Common Quick Fixes

1. **Service not responding**: Restart the service
2. **API key errors**: Check environment variables
3. **Memory issues**: Reduce batch size or chunk size
4. **Connection errors**: Verify service dependencies are running

## Installation Issues

### Problem: Dependencies Not Installing

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement
```

**Solutions:**
```bash
# Update pip and uv
pip install --upgrade pip uv

# Clear cache and reinstall
uv cache clean
uv sync --reinstall

# Check Python version (requires 3.11+)
python --version

# Install with verbose output
uv sync --verbose
```

### Problem: Import Errors

**Symptoms:**
```python
ModuleNotFoundError: No module named 'rag_engine'
```

**Solutions:**
```bash
# Ensure you're in the right directory
cd /path/to/rag-engine

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install in development mode
uv pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/rag-engine/src"
```

### Problem: Docker Build Failures

**Symptoms:**
```
ERROR: failed to solve: process "/bin/sh -c uv sync" did not complete successfully
```

**Solutions:**
```bash
# Clear Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t rag-engine .

# Check Docker resources
docker system df

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 8GB+
```

## Configuration Problems

### Problem: API Keys Not Working

**Symptoms:**
```
Error: 401 Unauthorized - Invalid API key
```

**Solutions:**
```bash
# Check environment variables
echo $GOOGLE_API_KEY
env | grep API_KEY

# Verify API key format
# Google AI: Should start with "AI..."
# OpenAI: Should start with "sk-..."

# Test API key directly
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     https://generativelanguage.googleapis.com/v1/models

# Set in Docker Compose
# docker-compose.yml:
environment:
  - GOOGLE_API_KEY=${GOOGLE_API_KEY}

# Set in Kubernetes
kubectl create secret generic rag-engine-secrets \
  --from-literal=GOOGLE_API_KEY=your_key_here \
  -n rag-engine
```

### Problem: Configuration File Not Found

**Symptoms:**
```
FileNotFoundError: Configuration file not found
```

**Solutions:**
```bash
# Check file exists
ls -la config/

# Create from example
cp config/example_config.yaml config/config.development.yaml

# Set config path
export RAG_CONFIG_PATH=/path/to/config.yaml

# Use absolute path in code
config_manager = ConfigurationManager(config_path="/absolute/path/config.yaml")
```

### Problem: Invalid Configuration Values

**Symptoms:**
```
ValidationError: Invalid configuration
```

**Solutions:**
```python
# Validate configuration
from rag_engine.core.config import ConfigurationManager

config_manager = ConfigurationManager()
try:
    config = config_manager.load_config()
    print("✅ Configuration valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")

# Check required fields
required_fields = ["llm_provider", "embedding_provider", "vector_store"]
for field in required_fields:
    if not hasattr(config, field):
        print(f"Missing required field: {field}")
```

## API and Authentication

### Problem: API Server Won't Start

**Symptoms:**
```
Error: Address already in use
```

**Solutions:**
```bash
# Check what's using the port
lsof -i :8000
netstat -tulpn | grep :8000

# Kill process using port
sudo kill -9 <PID>

# Use different port
export API_PORT=8001
# or in config:
api:
  port: 8001

# Check if service is already running
ps aux | grep rag-engine
```

### Problem: CORS Errors

**Symptoms:**
```
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:3000' 
has been blocked by CORS policy
```

**Solutions:**
```yaml
# config.yaml
api:
  cors_origins: 
    - "http://localhost:3000"
    - "https://yourdomain.com"
  cors_methods: ["GET", "POST", "PUT", "DELETE"]
  cors_headers: ["Content-Type", "Authorization"]

# Or allow all (development only)
api:
  cors_origins: ["*"]
```

### Problem: Rate Limiting Issues

**Symptoms:**
```
HTTP 429: Too Many Requests
```

**Solutions:**
```yaml
# Increase rate limits
api:
  rate_limit:
    requests_per_minute: 200
    burst_size: 50

# Disable rate limiting (development)
api:
  rate_limit:
    enabled: false

# Check current limits
curl -I http://localhost:8000/api/v1/health
# Look for X-RateLimit-* headers
```

## Document Indexing Issues

### Problem: Documents Not Indexing

**Symptoms:**
```
Error: Failed to index documents
```

**Solutions:**
```python
# Check document format
documents = [
    Document(
        content="Your content here",  # Must not be empty
        metadata={"source": "test"},  # Optional but recommended
        doc_id="unique_id"           # Optional
    )
]

# Validate documents before indexing
for doc in documents:
    if not doc.content.strip():
        print(f"Empty document: {doc.doc_id}")
    if len(doc.content) < 10:
        print(f"Very short document: {doc.doc_id}")

# Check indexing result
result = engine.add_documents(documents)
if not result:
    print("Indexing failed")
    # Check logs for details
```

### Problem: Chunk Size Issues

**Symptoms:**
```
Warning: Document chunks are too large/small
```

**Solutions:**
```yaml
# Adjust chunk size based on content type
indexing:
  # For short documents (tweets, Q&A)
  chunk_size: 300
  chunk_overlap: 50
  
  # For medium documents (articles)
  chunk_size: 1000
  chunk_overlap: 200
  
  # For long documents (books, papers)
  chunk_size: 1500
  chunk_overlap: 300

# Dynamic chunking based on content
chunk_size: auto  # Automatically determine optimal size
```

### Problem: Vector Store Connection Failed

**Symptoms:**
```
ConnectionError: Failed to connect to Chroma
```

**Solutions:**
```bash
# Check if Chroma is running
curl http://localhost:8001/api/v1/heartbeat

# Start Chroma (Docker)
docker run -p 8001:8000 chromadb/chroma:latest

# Check Docker Compose
docker-compose ps chroma
docker-compose logs chroma

# Verify configuration
vector_store:
  provider: "chroma"
  host: "localhost"  # or "chroma" in Docker Compose
  port: 8001
```

## Query Processing Problems

### Problem: No Relevant Documents Found

**Symptoms:**
```
Response: "I couldn't find any relevant information"
```

**Solutions:**
```python
# Check if documents are indexed
system_info = engine.get_system_info()
print(f"Indexed documents: {system_info['stats']['indexed_documents']}")

# Lower similarity threshold
config.score_threshold = 0.5  # Default is usually 0.7

# Increase retrieval count
response = engine.query("your question", k=10)

# Try different query strategies
response = engine.query("your question", strategy="multi_query")

# Check document content matches query domain
# If asking about "Python programming" but documents are about "cooking",
# no relevant documents will be found
```

### Problem: Poor Quality Responses

**Symptoms:**
- Irrelevant answers
- Hallucinated information
- Inconsistent responses

**Solutions:**
```yaml
# Improve retrieval quality
retrieval:
  k: 5                    # Retrieve more documents
  score_threshold: 0.8    # Higher threshold for relevance
  enable_reranking: true  # Use reranking for better results

# Improve generation quality
llm:
  temperature: 0.1        # Lower temperature for more factual responses
  model: "gemini-2.0-flash-lite" # Use higher quality model
  max_tokens: 500         # Limit response length

# Use better query strategies
query_processing:
  strategies: ["rag_fusion", "multi_query"]
  enable_query_expansion: true
```

### Problem: Slow Query Processing

**Symptoms:**
- High response times
- Timeouts

**Solutions:**
```yaml
# Optimize retrieval
retrieval:
  k: 3                    # Retrieve fewer documents
  enable_reranking: false # Disable reranking for speed
  
# Optimize generation
llm:
  model: "gemini-2.0-flash-lite" # Use faster model
  max_tokens: 200           # Shorter responses
  
# Enable caching
cache:
  enabled: true
  ttl: 3600
  
# Increase timeouts
api:
  timeout: 60
```

## Performance Issues

### Problem: High Memory Usage

**Symptoms:**
```
MemoryError: Unable to allocate memory
```

**Solutions:**
```yaml
# Reduce batch sizes
indexing:
  batch_size: 10          # Smaller batches
  
embedding:
  batch_size: 50          # Smaller embedding batches
  
# Reduce chunk size
indexing:
  chunk_size: 500         # Smaller chunks use less memory
  
# Enable streaming for large files
indexing:
  enable_streaming: true
  
# Set memory limits
memory:
  max_memory_usage: "2GB"
  gc_threshold: 0.7       # Trigger garbage collection earlier
```

### Problem: High CPU Usage

**Symptoms:**
- Slow processing
- High CPU utilization

**Solutions:**
```yaml
# Reduce concurrent processing
concurrency:
  max_concurrent_requests: 10
  max_workers: 2
  
# Optimize model settings
llm:
  model: "gemini-2.0-flash-lite"  # Faster model
  
# Enable GPU if available
gpu:
  enabled: true
  device: "cuda:0"
  
# Use connection pooling
connections:
  pool_size: 5
  max_connections: 20
```

### Problem: Slow Indexing

**Symptoms:**
- Long indexing times
- Timeouts during document processing

**Solutions:**
```python
# Process documents in smaller batches
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    engine.add_documents(batch)
    print(f"Processed batch {i//batch_size + 1}")

# Use async processing
import asyncio

async def index_documents_async(documents):
    tasks = []
    for doc in documents:
        task = asyncio.create_task(process_document(doc))
        tasks.append(task)
    return await asyncio.gather(*tasks)

# Optimize chunk size
config.chunk_size = 800  # Balance between speed and quality
```

## Memory and Resource Problems

### Problem: Out of Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
```bash
# Check available memory
free -h
docker stats

# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Reduce memory usage in config
memory:
  max_memory_usage: "1GB"
  enable_memory_monitoring: true
  
# Use memory-efficient settings
indexing:
  chunk_size: 500
  batch_size: 5
  enable_streaming: true
```

### Problem: Disk Space Issues

**Symptoms:**
```
OSError: No space left on device
```

**Solutions:**
```bash
# Check disk usage
df -h
docker system df

# Clean up Docker
docker system prune -a
docker volume prune

# Clean up logs
find /var/log -name "*.log" -mtime +7 -delete

# Configure log rotation
logging:
  max_file_size: "100MB"
  backup_count: 5
  
# Use external storage for vector database
vector_store:
  persist_directory: "/external/storage/chroma"
```

### Problem: Network Connectivity Issues

**Symptoms:**
```
ConnectionError: Failed to connect to external service
```

**Solutions:**
```bash
# Test connectivity
curl -I https://generativelanguage.googleapis.com
ping google.com

# Check DNS resolution
nslookup generativelanguage.googleapis.com

# Configure proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Set timeouts
llm:
  timeout: 60
  max_retries: 3
  
# Use connection pooling
connections:
  pool_size: 10
  keep_alive: true
```

## Deployment Issues

### Problem: Docker Container Won't Start

**Symptoms:**
```
Container exits immediately with code 1
```

**Solutions:**
```bash
# Check container logs
docker logs <container_id>

# Run container interactively
docker run -it rag-engine:latest /bin/bash

# Check Dockerfile
# Ensure WORKDIR is set correctly
# Ensure all dependencies are installed
# Ensure proper user permissions

# Common fixes:
# 1. Fix file permissions
RUN chown -R raguser:raguser /app

# 2. Install missing dependencies
RUN apt-get update && apt-get install -y curl

# 3. Set proper environment
ENV PYTHONPATH=/app/src
```

### Problem: Kubernetes Pods Failing

**Symptoms:**
```
Pod status: CrashLoopBackOff
```

**Solutions:**
```bash
# Check pod status
kubectl get pods -n rag-engine
kubectl describe pod <pod-name> -n rag-engine

# Check logs
kubectl logs <pod-name> -n rag-engine

# Check resource limits
kubectl describe pod <pod-name> -n rag-engine | grep -A 5 "Limits"

# Common issues:
# 1. Insufficient resources
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# 2. Missing secrets
kubectl get secrets -n rag-engine

# 3. ConfigMap issues
kubectl get configmap -n rag-engine
```

### Problem: Load Balancer Issues

**Symptoms:**
- Uneven load distribution
- Some requests failing

**Solutions:**
```yaml
# Configure session affinity
service:
  sessionAffinity: ClientIP
  
# Configure health checks
livenessProbe:
  httpGet:
    path: /health
    port: 8089
  initialDelaySeconds: 30
  periodSeconds: 10
  
readinessProbe:
  httpGet:
    path: /health
    port: 8089
  initialDelaySeconds: 10
  periodSeconds: 5

# Configure nginx upstream
upstream rag_engine {
    least_conn;
    server rag-engine-1:8000 max_fails=3 fail_timeout=30s;
    server rag-engine-2:8000 max_fails=3 fail_timeout=30s;
}
```

## Monitoring and Logging

### Problem: No Metrics Available

**Symptoms:**
- Empty Grafana dashboards
- No Prometheus metrics

**Solutions:**
```yaml
# Enable monitoring
monitoring:
  enabled: true
  metrics_port: 8089
  
# Check metrics endpoint
curl http://localhost:8089/metrics

# Verify Prometheus configuration
# prometheus.yml:
scrape_configs:
  - job_name: 'rag-engine'
    static_configs:
      - targets: ['rag-engine:8089']
    metrics_path: '/metrics'
```

### Problem: Log Files Too Large

**Symptoms:**
- Disk space issues
- Slow log processing

**Solutions:**
```yaml
# Configure log rotation
logging:
  max_file_size: "100MB"
  backup_count: 5
  compression: true
  
# Use structured logging
logging:
  format: "json"
  level: "INFO"  # Reduce verbosity
  
# Configure log cleanup
# Add to crontab:
0 2 * * * find /app/logs -name "*.log" -mtime +7 -delete
```

### Problem: Missing Log Entries

**Symptoms:**
- Gaps in logs
- Missing error information

**Solutions:**
```yaml
# Increase log level temporarily
logging:
  level: "DEBUG"
  
# Enable request logging
api:
  log_requests: true
  log_responses: false  # Don't log response content
  
# Check log file permissions
ls -la /app/logs/

# Ensure proper log configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/rag-engine.log'),
        logging.StreamHandler()
    ]
)
```

## Frequently Asked Questions

### Q: Which LLM provider should I use?

**A:** Choose based on your needs:
- **Google Gemini**: Good balance of quality and cost, fast inference
- **OpenAI GPT**: High quality, good for complex reasoning
- **Anthropic Claude**: Good for safety-critical applications
- **Local models**: For privacy-sensitive applications

### Q: How do I improve response quality?

**A:** Try these approaches:
1. **Better documents**: Ensure high-quality, relevant source material
2. **Optimize chunking**: Adjust chunk size for your content type
3. **Use reranking**: Enable document reranking for better relevance
4. **Query strategies**: Use multi-query or RAG-fusion strategies
5. **Model selection**: Use higher-quality models like GPT-4 or Gemini Pro

### Q: How many documents can the system handle?

**A:** Depends on your setup:
- **Memory-based**: 10K-100K documents
- **Disk-based**: 100K-1M+ documents
- **Cloud vector stores**: Millions of documents

Scale by:
- Using managed vector stores (Pinecone, Weaviate Cloud)
- Implementing document partitioning
- Using hierarchical indexing strategies

### Q: How do I handle different document types?

**A:** The system supports various formats:
```python
# Text files
doc = Document(content=text_content, metadata={"type": "text"})

# JSON documents
doc = Document(content=json.dumps(data), metadata={"type": "json"})

# Web pages (after scraping)
doc = Document(content=scraped_text, metadata={"type": "web", "url": url})

# For PDFs, Word docs, etc., use preprocessing:
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
pages = loader.load()
```

### Q: How do I secure the API in production?

**A:** Implement these security measures:
1. **API Authentication**: Use API keys or JWT tokens
2. **HTTPS**: Enable TLS/SSL encryption
3. **Rate Limiting**: Prevent abuse
4. **CORS**: Configure allowed origins
5. **Input Validation**: Sanitize all inputs
6. **Network Security**: Use firewalls and VPCs

### Q: Can I use custom models?

**A:** Yes, you can integrate custom models:
```python
# Custom LLM provider
class CustomLLMProvider(BaseLLMProvider):
    def generate(self, prompt: str) -> str:
        # Your custom model logic
        return response

# Custom embedding provider
class CustomEmbeddingProvider(BaseEmbeddingProvider):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Your custom embedding logic
        return embeddings
```

### Q: How do I monitor system performance?

**A:** Use the built-in monitoring:
1. **Metrics**: Prometheus metrics at `/metrics`
2. **Health checks**: Health endpoint at `/health`
3. **Logging**: Structured logs with performance data
4. **Dashboards**: Grafana dashboards for visualization

### Q: What's the best chunk size for my documents?

**A:** Depends on your content and use case:
- **Short Q&A**: 300-500 characters
- **Articles/Blogs**: 800-1200 characters
- **Technical docs**: 1000-1500 characters
- **Books/Papers**: 1200-2000 characters

Test different sizes and measure retrieval quality.

### Q: How do I handle multilingual content?

**A:** Consider these approaches:
1. **Language-specific models**: Use models trained on your target language
2. **Translation**: Translate queries/documents to a common language
3. **Multilingual embeddings**: Use models like multilingual-E5
4. **Language detection**: Route queries based on detected language

### Q: Can I run this without internet access?

**A:** Yes, for offline deployment:
1. **Local LLM**: Use Ollama or similar local models
2. **Local embeddings**: Use Hugging Face sentence transformers
3. **Local vector store**: Use Chroma in persistent mode
4. **No external APIs**: Configure all providers to use local services

### Q: How do I backup and restore data?

**A:** Backup strategies:
```bash
# Backup Chroma database
tar -czf chroma-backup.tar.gz /path/to/chroma/data

# Backup configuration
tar -czf config-backup.tar.gz config/

# Backup using Velero (Kubernetes)
velero backup create rag-engine-backup --include-namespaces rag-engine

# Restore
tar -xzf chroma-backup.tar.gz -C /path/to/restore/
```

### Q: How do I scale horizontally?

**A:** Scaling approaches:
1. **Stateless API**: Multiple API server instances
2. **Load balancing**: Distribute requests across instances
3. **Shared storage**: Use external vector stores and caches
4. **Kubernetes**: Use HPA for automatic scaling
5. **Microservices**: Split into separate services (indexing, retrieval, generation)

For more specific issues not covered here, check the logs, enable debug mode, and consult the API documentation.