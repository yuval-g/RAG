# Docker Deployment Guide

This guide covers deploying the RAG Engine using Docker and Docker Compose.

## Quick Start

### Development Environment

```bash
# Clone repository
git clone <repository-url>
cd rag-engine

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Start development environment
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f rag-engine
```

### Production Environment

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Or use deployment script
./deployment/scripts/deploy.sh production
```

## Docker Images

### Base Images

- **Production**: `python:3.13-slim` - Minimal production image
- **Development**: `python:3.13-slim` - Development image with tools

### Multi-stage Build

The production Dockerfile uses multi-stage builds:

1. **Builder stage**: Installs dependencies and builds the application
2. **Production stage**: Copies only necessary files for runtime

### Image Optimization

- Uses `.dockerignore` to exclude unnecessary files
- Leverages Docker layer caching
- Runs as non-root user for security
- Includes health checks

## Service Architecture

### Core Services

```yaml
services:
  rag-engine:     # Main API server
  chroma:         # Vector database
  redis:          # Caching layer
  nginx:          # Load balancer
  prometheus:     # Metrics collection
  grafana:        # Monitoring dashboard
```

### Network Configuration

All services run on a custom bridge network `rag-network` for:
- Service discovery by name
- Network isolation
- Internal communication

### Volume Management

```yaml
volumes:
  rag_data:       # Application data
  rag_logs:       # Application logs
  chroma_data:    # Vector database storage
  redis_data:     # Cache storage
  prometheus_data: # Metrics storage
  grafana_data:   # Dashboard configuration
```

## Environment Configuration

### Environment Variables

#### Core Configuration
```bash
ENVIRONMENT=production          # Environment type
LOG_LEVEL=INFO                 # Logging level
WORKERS=4                      # Number of worker processes
```

#### API Keys
```bash
GOOGLE_API_KEY=your_key        # Google AI API key
COHERE_API_KEY=your_key        # Cohere API key (optional)
```

#### Service Configuration
```bash
CHROMA_HOST=chroma             # Chroma service hostname
CHROMA_PORT=8001               # Chroma service port
REDIS_URL=redis://redis:6379   # Redis connection URL
```

#### Security
```bash
GRAFANA_PASSWORD=secure_pass   # Grafana admin password
JWT_SECRET=your_secret         # JWT signing secret
```

### Configuration Files

Mount configuration files as volumes:

```yaml
volumes:
  - ./config:/app/config:ro     # Read-only config mount
  - ./deployment/nginx.conf:/etc/nginx/nginx.conf:ro
```

## Service Details

### RAG Engine Service

```yaml
rag-engine:
  build:
    context: .
    dockerfile: Dockerfile
  ports:
    - "8000:8000"    # API port
    - "8089:8089"    # Health/metrics port
  environment:
    - ENVIRONMENT=production
    - GOOGLE_API_KEY=${GOOGLE_API_KEY}
  volumes:
    - rag_data:/app/data
    - rag_logs:/app/logs
  depends_on:
    - chroma
    - redis
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8089/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Chroma Vector Database

```yaml
chroma:
  image: chromadb/chroma:latest
  ports:
    - "8001:8000"
  environment:
    - CHROMA_SERVER_HOST=0.0.0.0
    - PERSIST_DIRECTORY=/chroma/chroma
  volumes:
    - chroma_data:/chroma/chroma
  restart: unless-stopped
```

### Redis Cache

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  command: redis-server --appendonly yes
  restart: unless-stopped
```

### Nginx Load Balancer

```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./deployment/nginx.conf:/etc/nginx/nginx.conf:ro
    - ./deployment/ssl:/etc/nginx/ssl:ro
  depends_on:
    - rag-engine
  restart: unless-stopped
```

## Development Workflow

### Hot Reloading

Development configuration includes:

```yaml
rag-engine:
  build:
    dockerfile: Dockerfile.dev
  volumes:
    - .:/app                    # Mount source code
  command: ["uv", "run", "python", "-m", "src.rag_engine.api.main", "--reload"]
```

### Debugging

Access container for debugging:

```bash
# Execute shell in container
docker-compose exec rag-engine bash

# View real-time logs
docker-compose logs -f rag-engine

# Check container stats
docker stats

# Inspect container
docker-compose exec rag-engine python -c "
from src.rag_engine.core.engine import RAGEngine
engine = RAGEngine()
print(engine.get_system_info())
"
```

### Development Tools

Development image includes additional tools:

- `vim` - Text editor
- `htop` - Process monitor
- `curl` - HTTP client
- Development dependencies

## Production Optimizations

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### Security Hardening

1. **Non-root User**:
   ```dockerfile
   RUN groupadd -r raguser && useradd -r -g raguser raguser
   USER raguser
   ```

2. **Read-only Filesystem**:
   ```yaml
   security_opt:
     - no-new-privileges:true
   read_only: true
   tmpfs:
     - /tmp
     - /var/tmp
   ```

3. **Network Security**:
   ```yaml
   networks:
     rag-network:
       driver: bridge
       ipam:
         config:
           - subnet: 172.20.0.0/16
   ```

### Performance Tuning

1. **Memory Management**:
   ```yaml
   environment:
     - PYTHONUNBUFFERED=1
     - PYTHONDONTWRITEBYTECODE=1
   ```

2. **Connection Pooling**:
   ```yaml
   environment:
     - MAX_CONNECTIONS=100
     - POOL_SIZE=20
   ```

## Monitoring and Logging

### Prometheus Configuration

```yaml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"
  volumes:
    - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    - prometheus_data:/prometheus
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.retention.time=200h'
```

### Grafana Setup

```yaml
grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
  volumes:
    - grafana_data:/var/lib/grafana
    - ./deployment/grafana:/etc/grafana/provisioning:ro
```

### Log Management

1. **Centralized Logging**:
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

2. **Log Rotation**:
   ```bash
   # Add to crontab
   0 2 * * * docker system prune -f --filter "until=24h"
   ```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup Chroma data
docker-compose exec -T chroma tar -czf - /chroma/chroma > "$BACKUP_DIR/chroma_$DATE.tar.gz"

# Backup Redis data
docker-compose exec -T redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Backup application data
docker run --rm -v rag_data:/data -v "$BACKUP_DIR":/backup alpine tar -czf "/backup/app_data_$DATE.tar.gz" -C /data .

echo "Backup completed: $DATE"
```

### Restore Procedure

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

# Restore data
docker run --rm -v rag_data:/data -v "$(pwd)":/backup alpine tar -xzf "/backup/$BACKUP_FILE" -C /data

# Start services
docker-compose up -d

echo "Restore completed from: $BACKUP_FILE"
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Change ports in docker-compose.yml
   ports:
     - "8080:8000"  # Use different host port
   ```

2. **Permission Issues**:
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./data
   
   # Or use named volumes instead of bind mounts
   ```

3. **Memory Issues**:
   ```bash
   # Check container memory usage
   docker stats
   
   # Increase memory limits
   deploy:
     resources:
       limits:
         memory: 8G
   ```

4. **Network Issues**:
   ```bash
   # Check network connectivity
   docker-compose exec rag-engine ping chroma
   
   # Inspect network
   docker network inspect rag-engine_rag-network
   ```

### Debug Commands

```bash
# Container inspection
docker-compose exec rag-engine env
docker-compose exec rag-engine ps aux
docker-compose exec rag-engine netstat -tulpn

# Service health
curl http://localhost:8000/health
curl http://localhost:8001/api/v1/heartbeat
curl http://localhost:9090/targets

# Log analysis
docker-compose logs --tail=100 rag-engine
docker-compose logs --since=1h chroma
```

### Performance Monitoring

```bash
# Real-time stats
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Resource usage over time
docker-compose exec prometheus promtool query instant 'rate(container_cpu_usage_seconds_total[5m])'
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  rag-engine:
    deploy:
      replicas: 3
    
  nginx:
    volumes:
      - ./deployment/nginx-scaled.conf:/etc/nginx/nginx.conf:ro
```

### Load Balancing Configuration

```nginx
upstream rag_engine {
    least_conn;
    server rag-engine_1:8000;
    server rag-engine_2:8000;
    server rag-engine_3:8000;
}
```

## Security Best Practices

### Container Security

1. **Use Official Images**:
   ```yaml
   image: python:3.13-slim  # Official Python image
   ```

2. **Scan for Vulnerabilities**:
   ```bash
   docker scan rag-engine:latest
   ```

3. **Limit Capabilities**:
   ```yaml
   cap_drop:
     - ALL
   cap_add:
     - NET_BIND_SERVICE
   ```

### Network Security

1. **Internal Networks**:
   ```yaml
   networks:
     internal:
       internal: true  # No external access
     external:
       # External access allowed
   ```

2. **Firewall Rules**:
   ```bash
   # Allow only necessary ports
   ufw allow 80/tcp
   ufw allow 443/tcp
   ufw deny 8000/tcp  # Block direct API access
   ```

### Secrets Management

1. **Docker Secrets**:
   ```yaml
   secrets:
     google_api_key:
       file: ./secrets/google_api_key.txt
   
   services:
     rag-engine:
       secrets:
         - google_api_key
   ```

2. **External Secret Management**:
   ```bash
   # Use HashiCorp Vault, AWS Secrets Manager, etc.
   export GOOGLE_API_KEY=$(vault kv get -field=key secret/rag-engine/google)
   ```

## Maintenance

### Regular Tasks

1. **Update Images**:
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

2. **Clean Up**:
   ```bash
   docker system prune -f
   docker volume prune -f
   ```

3. **Health Checks**:
   ```bash
   ./deployment/scripts/health-check.sh
   ```

### Automated Maintenance

```bash
#!/bin/bash
# maintenance.sh

# Update and restart services
docker-compose pull
docker-compose up -d

# Clean up old images
docker image prune -f --filter "until=168h"

# Backup data
./deployment/scripts/backup.sh

# Check health
./deployment/scripts/health-check.sh

echo "Maintenance completed: $(date)"
```

Add to crontab:
```bash
# Weekly maintenance
0 2 * * 0 /path/to/maintenance.sh >> /var/log/rag-engine-maintenance.log 2>&1
```