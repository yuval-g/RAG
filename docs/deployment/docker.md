# Docker Deployment

This guide covers deploying the RAG System using Docker and Docker Compose for both development and production environments.

## Overview

The RAG System can be deployed using Docker containers for easy setup, consistent environments, and simplified management across different stages.

## Prerequisites

*   **Docker**: Ensure Docker Desktop (for Windows/macOS) or Docker Engine (for Linux) is installed and running.
*   **Docker Compose**: Usually comes bundled with Docker Desktop. For Linux, install it separately if needed.
*   **Git**: For cloning the RAG System repository.

## Quick Start

### Development Environment

For local development with hot-reloading and debugging capabilities:

1.  **Clone the repository:**
```bash
git clone <repository-url>
cd rag-engine # Or your project root directory
```

2.  **Copy environment file:**
```bash
cp .env.example .env
```
Edit the `.env` file to configure necessary API keys (e.g., `GOOGLE_API_KEY`, `OPENAI_API_KEY`) and other environment-specific settings.

3.  **Start the development environment:**
```bash
docker-compose up -d
```
This command builds the images (if not already built) and starts the services in detached mode.

4.  **Check status and logs:**
```bash
docker-compose ps
docker-compose logs -f rag-engine
```

### Production Environment

For a production-ready deployment, use the dedicated production Docker Compose file:

1.  **Start the production environment:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```
Alternatively, you can use the provided deployment script:
```bash
./deployment/scripts/deploy.sh production
```

## Docker Images

### Base Images

*   **Production**: Uses `python:3.13-slim` for a minimal and secure production image.
*   **Development**: Also based on `python:3.13-slim`, but includes additional development tools and dependencies.

### Multi-stage Build

The `Dockerfile` (for production) utilizes multi-stage builds to create optimized and smaller images:

1.  **Builder Stage**: Installs all application dependencies and builds any necessary artifacts.
2.  **Production Stage**: Copies only the essential runtime files from the builder stage, resulting in a lean final image.

### Image Optimization

*   **`.dockerignore`**: Excludes unnecessary files and directories from the build context.
*   **Layer Caching**: Leverages Docker's build cache for faster subsequent builds.
*   **Non-root User**: Runs the application as a non-root user for enhanced security.
*   **Health Checks**: Includes `HEALTHCHECK` instructions to verify container health.

## Service Architecture

The Docker Compose setup defines a multi-service architecture:

*   `rag-engine`: The main RAG API server.
*   `chroma`: The ChromaDB vector database for document storage.
*   `redis`: A Redis instance for caching and session management.
*   `nginx`: An Nginx reverse proxy/load balancer (primarily for production).
*   `prometheus`: For metrics collection and monitoring.
*   `grafana`: For visualizing metrics and dashboards.

### Network Configuration

All services are connected via a custom bridge network (e.g., `rag-network`), enabling:

*   **Service Discovery**: Services can communicate using their service names (e.g., `rag-engine` can connect to `chroma`).
*   **Network Isolation**: Services are isolated from the host network unless explicitly exposed.
*   **Internal Communication**: Optimized for inter-service communication.

### Volume Management

Persistent data is managed using Docker volumes to ensure data persistence across container restarts:

*   `rag_data`: For application-specific data.
*   `rag_logs`: For application logs.
*   `chroma_data`: For the ChromaDB persistent storage.
*   `redis_data`: For Redis data persistence.
*   `prometheus_data`: For Prometheus metrics storage.
*   `grafana_data`: For Grafana dashboards and configuration.

## Environment Configuration

### Environment Variables

Environment variables are crucial for configuring the RAG Engine within Docker containers. They can be set directly in `docker-compose.yml` or loaded from a `.env` file.

#### Core Configuration

```bash
ENVIRONMENT=production          # Environment type (development, testing, production)
LOG_LEVEL=INFO                 # Logging level (DEBUG, INFO, WARNING, ERROR)
WORKERS=4                      # Number of Uvicorn worker processes for the API server
```

#### API Keys

```bash
GOOGLE_API_KEY=your_google_key # Your Google AI API key
OPENAI_API_KEY=your_openai_key # Your OpenAI API key (if using OpenAI models)
COHERE_API_KEY=your_cohere_key # Your Cohere API key (optional, for reranking)
```

#### Service Configuration

```bash
CHROMA_HOST=chroma             # Hostname of the Chroma service within the Docker network
CHROMA_PORT=8001               # Port of the Chroma service
REDIS_URL=redis://redis:6379   # Connection URL for the Redis service
```

#### Security-related Variables

```bash
GRAFANA_PASSWORD=secure_password # Admin password for Grafana
JWT_SECRET=your_jwt_secret       # Secret for JWT signing (if authentication is enabled)
```

### Configuration Files

Mount configuration files as read-only volumes to provide dynamic configuration without rebuilding images:

```yaml
volumes:
  - ./config:/app/config:ro     # Mounts the host's config directory to /app/config in the container
  - ./deployment/nginx.conf:/etc/nginx/nginx.conf:ro # Mounts Nginx configuration
```

## Service Details (Example `docker-compose.yml` Snippets)

### RAG Engine Service

```yaml
rag-engine:
  build:
    context: .
    dockerfile: Dockerfile # Or Dockerfile.dev for development
  ports:
    - "8000:8000"    # Exposes the API port (host:container)
    - "8089:8089"    # Exposes the Health/Metrics port
  environment:
    - ENVIRONMENT=production
    - GOOGLE_API_KEY=${GOOGLE_API_KEY} # Loaded from .env file
  volumes:
    - rag_data:/app/data
    - rag_logs:/app/logs
  depends_on:
    - chroma # Ensures Chroma starts before rag-engine
    - redis  # Ensures Redis starts before rag-engine
  restart: unless-stopped # Always restart unless explicitly stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8089/health"] # Health check command
    interval: 30s
    timeout: 10s
    retries: 3
```

### Chroma Vector Database

```yaml
chroma:
  image: chromadb/chroma:latest # Official ChromaDB image
  ports:
    - "8001:8000" # Exposes ChromaDB's default port
  environment:
    - CHROMA_SERVER_HOST=0.0.0.0
    - PERSIST_DIRECTORY=/chroma/chroma # Directory for persistent data within container
  volumes:
    - chroma_data:/chroma/chroma # Mounts persistent volume
  restart: unless-stopped
```

### Redis Cache

```yaml
redis:
  image: redis:7-alpine # Lightweight Redis image
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data # Mounts persistent volume for Redis data
  command: redis-server --appendonly yes # Enables AOF persistence
  restart: unless-stopped
```

### Nginx Load Balancer (Production Example)

```yaml
nginx:
  image: nginx:alpine # Lightweight Nginx image
  ports:
    - "80:80"  # HTTP
    - "443:443" # HTTPS
  volumes:
    - ./deployment/nginx.conf:/etc/nginx/nginx.conf:ro # Mounts Nginx config
    - ./deployment/ssl:/etc/nginx/ssl:ro # Mounts SSL certificates
  depends_on:
    - rag-engine
  restart: unless-stopped
```

## Development Workflow

### Hot Reloading

The development `docker-compose.yml` (or `Dockerfile.dev`) is configured for hot-reloading:

```yaml
rag-engine:
  build:
    dockerfile: Dockerfile.dev
  volumes:
    - .:/app # Mounts the host's project directory into the container
  command: ["uv", "run", "python", "-m", "src.rag_engine.api.main", "--reload"] # Starts server with reload
```

This setup allows code changes on the host machine to be immediately reflected in the running container.

### Debugging

*   **Execute shell in container:**
```bash
docker-compose exec rag-engine bash
```
*   **View real-time logs:**
```bash
docker-compose logs -f rag-engine
```
*   **Check container stats:**
```bash
docker stats
```
*   **Inspect container (e.g., run Python code inside):**
```bash
docker-compose exec rag-engine python -c "
from src.rag_engine.core.engine import RAGEngine
engine = RAGEngine()
print(engine.get_system_info())
"
```

### Development Tools

The `Dockerfile.dev` includes additional tools useful for development and debugging:

*   `vim` - Text editor
*   `htop` - Process monitor
*   `curl` - HTTP client
*   All development dependencies required for testing and linting.

## Production Optimizations

### Resource Limits

Define CPU and memory limits to prevent resource exhaustion and ensure stable operation:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0' # Max 2 CPU cores
      memory: 4G  # Max 4 GB RAM
    reservations:
      cpus: '1.0' # Reserve 1 CPU core
      memory: 2G  # Reserve 2 GB RAM
```

### Security Hardening

1.  **Non-root User**: Run containers with a dedicated non-root user.
```dockerfile
RUN groupadd -r raguser && useradd -r -g raguser raguser
USER raguser
```

2.  **Read-only Filesystem**: Mount the container's root filesystem as read-only, allowing writes only to specific volumes or `tmpfs`.
```yaml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

3.  **Network Security**: Configure custom networks with specific subnets and isolation.
```yaml
networks:
  rag-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Performance Tuning

1.  **Python Environment Variables**: Optimize Python's behavior for containerized environments.
```yaml
environment:
  - PYTHONUNBUFFERED=1 # Ensures stdout/stderr are unbuffered
  - PYTHONDONTWRITEBYTECODE=1 # Prevents .pyc file creation
```

2.  **Connection Pooling**: Configure connection pool sizes for external services.
```yaml
environment:
  - MAX_CONNECTIONS=100 # Example: Max connections for a database pool
  - POOL_SIZE=20      # Example: Initial pool size
```

## Monitoring and Logging

### Prometheus Configuration

Integrate with Prometheus for time-series metrics collection:

```yaml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"
  volumes:
    - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml:ro # Mount Prometheus config
    - prometheus_data:/prometheus # Persistent storage for metrics
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.retention.time=200h' # Data retention
```

### Grafana Setup

Visualize metrics and create dashboards with Grafana:

```yaml
grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD} # Admin password
  volumes:
    - grafana_data:/var/lib/grafana # Persistent storage for Grafana data
    - ./deployment/grafana:/etc/grafana/provisioning:ro # Provisioning dashboards
```

### Log Management

1.  **Centralized Logging**: Configure Docker to send logs to a centralized logging solution.
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

2.  **Log Rotation**: Implement log rotation to prevent disk space issues.
```bash
# Example crontab entry for daily Docker system prune
0 2 * * * docker system prune -f --filter "until=24h"
```

## Backup and Recovery

Implement robust backup and recovery procedures for persistent data.

### Data Backup Script Example

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

mkdir -p "$BACKUP_DIR"

# Backup Chroma data
docker-compose exec -T chroma tar -czf - /chroma/chroma > "$BACKUP_DIR/chroma_$DATE.tar.gz"

# Backup Redis data
docker-compose exec -T redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Backup application data (assuming rag_data volume)
docker run --rm -v rag_data:/data -v "$BACKUP_DIR":/backup alpine tar -czf "/backup/app_data_$DATE.tar.gz" -C /data .

echo "Backup completed: $DATE"
```

### Restore Procedure Example

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop services
docker-compose down

# Restore data (example for application data volume)
docker run --rm -v rag_data:/data -v "$(pwd)":/backup alpine tar -xzf "/backup/$BACKUP_FILE" -C /data

# For Chroma/Redis, you might need to copy the backup into their respective volumes
# and restart the services.

# Start services
docker-compose up -d

echo "Restore completed from: $BACKUP_FILE"
```

## Troubleshooting

### Common Issues

1.  **Port Conflicts**: Another process is already using the required port.
    *   **Solution**: Check port usage (`netstat -tulpn | grep :<port>`) and change ports in `docker-compose.yml`.

2.  **Permission Issues**: Incorrect file or directory permissions for mounted volumes.
    *   **Solution**: Fix volume permissions (`sudo chown -R <user>:<group> ./data`) or use named volumes.

3.  **Memory Issues**: Container running out of memory.
    *   **Solution**: Check container memory usage (`docker stats`) and increase memory limits in `docker-compose.yml`.

4.  **Network Issues**: Services cannot communicate with each other.
    *   **Solution**: Check network connectivity (`docker-compose exec <service> ping <other_service>`) and inspect Docker networks.

### Debug Commands

*   **Container inspection**: `docker-compose exec <service> env`, `docker-compose exec <service> ps aux`
*   **Service health**: `curl http://localhost:8000/health`, `curl http://localhost:8001/api/v1/heartbeat`
*   **Log analysis**: `docker-compose logs -f <service>`, `docker-compose logs --since=1h <service>`

## Scaling

### Horizontal Scaling with Docker Compose

For simple horizontal scaling, you can use `deploy.replicas` in a `docker-compose.override.yml`:

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  rag-engine:
    deploy:
      replicas: 3 # Scale rag-engine to 3 instances
    
  nginx:
    volumes:
      - ./deployment/nginx-scaled.conf:/etc/nginx/nginx.conf:ro # Update Nginx config for multiple backends
```

### Load Balancing Configuration (Nginx Example)

When scaling `rag-engine`, ensure your Nginx configuration is updated to load balance across all instances:

```nginx
upstream rag_engine {
    least_conn; # Distribute requests to the backend with the least active connections
    server rag-engine_1:8000;
    server rag-engine_2:8000;
    server rag-engine_3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://rag_engine;
        # ... other proxy settings
    }
}
```

## Security Best Practices

### Container Security

1.  **Use Official Images**: Always prefer official and slim base images.
2.  **Scan for Vulnerabilities**: Regularly scan your Docker images for known vulnerabilities.
3.  **Limit Capabilities**: Drop unnecessary Linux capabilities (`cap_drop: ALL`) and add only what's essential (`cap_add: NET_BIND_SERVICE`).

### Network Security

1.  **Internal Networks**: Use Docker's internal networks for inter-service communication to prevent external exposure.
2.  **Firewall Rules**: Implement host-level firewall rules to only expose necessary ports.

### Secrets Management

1.  **Docker Secrets**: Utilize Docker Swarm secrets for sensitive information in production.
2.  **External Secret Management**: Integrate with external secret management systems (e.g., HashiCorp Vault, AWS Secrets Manager).

## Maintenance

### Regular Tasks

1.  **Update Images**: Regularly pull the latest base images and rebuild your application images.
2.  **Clean Up**: Periodically prune unused Docker objects (containers, images, volumes, networks).
3.  **Health Checks**: Regularly verify the health of your running containers and services.

### Automated Maintenance (Example Crontab Entry)

```bash
#!/bin/bash
# maintenance.sh

# Update and restart services
docker-compose pull
docker-compose up -d

# Clean up old images (e.g., older than 7 days)
docker image prune -f --filter "until=168h"

# Backup data (if applicable)
# ./deployment/scripts/backup.sh

# Check health
# ./deployment/scripts/health-check.sh

echo "Maintenance completed: $(date)"
```

Add this script to your system's crontab for automated execution (e.g., weekly):

```bash
# Weekly maintenance at 2 AM on Sunday
0 2 * * 0 /path/to/your/maintenance.sh >> /var/log/rag-engine-maintenance.log 2>&1
```
