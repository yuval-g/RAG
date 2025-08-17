# RAG Engine Deployment Guide

This guide covers deploying the RAG Engine in various environments using Docker Compose and Kubernetes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Troubleshooting](#troubleshooting)
7. [Scaling and Performance](#scaling-and-performance)

## Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 8GB RAM, Recommended 16GB+ RAM
- **Storage**: Minimum 50GB, Recommended 100GB+ SSD
- **Network**: Stable internet connection for API calls

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- kubectl 1.24+ (for Kubernetes deployment)
- curl (for health checks)

### API Keys and Credentials

- Google AI API Key (for Gemini models)
- Optional: Cohere API Key (for reranking)
- SSL certificates (for production HTTPS)

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKERS=4

# API Keys
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # Optional

# Database Configuration
CHROMA_HOST=chroma
CHROMA_PORT=8001
REDIS_URL=redis://redis:6379

# Security
GRAFANA_PASSWORD=secure_password_here
JWT_SECRET=your_jwt_secret_here

# Performance
MAX_CONCURRENT_REQUESTS=100
CACHE_TTL=3600
```

### Configuration Files

The system uses YAML configuration files in the `config/` directory:

- `config.development.yaml` - Development settings
- `config.production.yaml` - Production settings
- `config.testing.yaml` - Testing settings

## Docker Compose Deployment

### Development Deployment

For development with hot reloading:

```bash
# Clone the repository
git clone <repository-url>
cd rag-engine

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Deploy development environment
./deployment/scripts/deploy.sh development

# Or manually:
docker-compose up -d
```

### Production Deployment

For production deployment:

```bash
# Deploy production environment
./deployment/scripts/deploy.sh production

# Or manually:
docker-compose -f docker-compose.prod.yml up -d
```

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| rag-engine | 8000 | Main API server |
| rag-engine | 8089 | Health and metrics |
| chroma | 8001 | Vector database |
| redis | 6379 | Caching layer |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboard |
| nginx | 80/443 | Load balancer |

### Health Checks

Verify deployment:

```bash
# Check service health
curl http://localhost:8000/health

# Check all services
docker-compose ps

# View logs
docker-compose logs -f rag-engine
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Ingress controller (nginx-ingress recommended)
- Storage classes configured

### Deployment Steps

1. **Prepare Secrets**:

```bash
# Create namespace
kubectl apply -f deployment/k8s/namespace.yaml

# Create secrets (update with your values)
kubectl create secret generic rag-engine-secrets \
  --from-literal=GOOGLE_API_KEY=your_api_key \
  --from-literal=GRAFANA_PASSWORD=your_password \
  -n rag-engine

# Create SSL certificates
kubectl create secret tls ssl-certs \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n rag-engine
```

2. **Deploy Services**:

```bash
# Deploy using script
./deployment/scripts/deploy.sh kubernetes

# Or manually:
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/redis-deployment.yaml
kubectl apply -f deployment/k8s/chroma-deployment.yaml
kubectl apply -f deployment/k8s/rag-engine-deployment.yaml
kubectl apply -f deployment/k8s/nginx-deployment.yaml
kubectl apply -f deployment/k8s/monitoring.yaml
```

3. **Verify Deployment**:

```bash
# Check pod status
kubectl get pods -n rag-engine

# Check services
kubectl get services -n rag-engine

# Check ingress
kubectl get ingress -n rag-engine

# View logs
kubectl logs -f deployment/rag-engine -n rag-engine
```

### Scaling

Scale the RAG Engine deployment:

```bash
# Scale to 5 replicas
kubectl scale deployment rag-engine --replicas=5 -n rag-engine

# Auto-scaling (HPA)
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-engine-hpa
  namespace: rag-engine
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-engine
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
```

## Monitoring and Observability

### Prometheus Metrics

The RAG Engine exposes metrics at `/metrics` endpoint:

- `rag_engine_requests_total` - Total API requests
- `rag_engine_request_duration_seconds` - Request duration
- `rag_engine_query_processing_time` - Query processing time
- `rag_engine_retrieval_documents_count` - Retrieved documents count
- `rag_engine_confidence_score` - Response confidence scores

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (Docker Compose) or through ingress (K8s):

- **RAG Engine Overview**: System health and performance
- **API Metrics**: Request rates, latency, errors
- **Resource Usage**: CPU, memory, disk usage
- **Query Analytics**: Query patterns and performance

### Log Aggregation

For production, consider integrating with:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Fluentd** for log collection
- **Jaeger** for distributed tracing

### Alerting

Configure alerts in Prometheus:

```yaml
# deployment/prometheus/rules/rag-engine.yml
groups:
- name: rag-engine
  rules:
  - alert: RAGEngineDown
    expr: up{job="rag-engine"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "RAG Engine is down"
      
  - alert: HighErrorRate
    expr: rate(rag_engine_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(rag_engine_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**:
   ```bash
   # Check logs
   docker-compose logs rag-engine
   kubectl logs deployment/rag-engine -n rag-engine
   
   # Check configuration
   docker-compose config
   kubectl describe deployment rag-engine -n rag-engine
   ```

2. **API Key Issues**:
   ```bash
   # Verify environment variables
   docker-compose exec rag-engine env | grep GOOGLE_API_KEY
   kubectl get secret rag-engine-secrets -n rag-engine -o yaml
   ```

3. **Database Connection Issues**:
   ```bash
   # Check Chroma connectivity
   curl http://localhost:8001/api/v1/heartbeat
   kubectl port-forward service/chroma-service 8001:8000 -n rag-engine
   ```

4. **Memory Issues**:
   ```bash
   # Check resource usage
   docker stats
   kubectl top pods -n rag-engine
   
   # Increase memory limits
   # Edit docker-compose.yml or k8s deployment
   ```

### Performance Tuning

1. **Optimize Chunk Size**:
   - Smaller chunks (300-500): Better precision, more API calls
   - Larger chunks (800-1200): Better context, fewer API calls

2. **Adjust Retrieval Parameters**:
   - `retrieval_k`: Number of documents to retrieve
   - `reranking`: Enable for better relevance

3. **Scale Components**:
   - Increase RAG Engine replicas for higher throughput
   - Scale Chroma for better vector search performance
   - Add Redis replicas for caching

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Docker Compose
docker-compose exec rag-engine python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.rag_engine.core.engine import RAGEngine
engine = RAGEngine()
print(engine.get_system_info())
"

# Kubernetes
kubectl exec -it deployment/rag-engine -n rag-engine -- python -c "..."
```

## Scaling and Performance

### Horizontal Scaling

1. **Load Balancer Configuration**:
   - Use nginx for load balancing
   - Configure health checks
   - Set appropriate timeouts

2. **Database Scaling**:
   - Chroma: Consider clustering for large datasets
   - Redis: Use Redis Cluster for high availability

3. **Caching Strategy**:
   - Cache frequent queries
   - Use Redis for session storage
   - Implement query result caching

### Vertical Scaling

1. **Resource Allocation**:
   ```yaml
   resources:
     requests:
       memory: "2Gi"
       cpu: "1000m"
     limits:
       memory: "4Gi"
       cpu: "2000m"
   ```

2. **JVM Tuning** (if using Java components):
   ```bash
   JAVA_OPTS="-Xmx4g -Xms2g -XX:+UseG1GC"
   ```

### Performance Monitoring

Monitor key metrics:

- **Throughput**: Requests per second
- **Latency**: P95, P99 response times
- **Error Rate**: 4xx, 5xx responses
- **Resource Usage**: CPU, memory, disk I/O
- **Queue Depth**: Pending requests

### Optimization Tips

1. **Query Optimization**:
   - Use appropriate query strategies
   - Optimize embedding models
   - Implement query caching

2. **Infrastructure**:
   - Use SSD storage for databases
   - Optimize network configuration
   - Use CDN for static assets

3. **Application**:
   - Implement connection pooling
   - Use async processing where possible
   - Optimize batch operations

## Security Considerations

### Network Security

- Use HTTPS in production
- Implement rate limiting
- Configure firewall rules
- Use VPC/private networks

### Authentication & Authorization

- Implement API key authentication
- Use JWT tokens for sessions
- Configure RBAC in Kubernetes
- Audit API access

### Data Security

- Encrypt data at rest
- Use secure communication channels
- Implement data retention policies
- Regular security updates

## Backup and Recovery

### Data Backup

1. **Chroma Database**:
   ```bash
   # Backup Chroma data
   docker-compose exec chroma tar -czf /backup/chroma-$(date +%Y%m%d).tar.gz /chroma/chroma
   ```

2. **Configuration**:
   ```bash
   # Backup configurations
   tar -czf config-backup-$(date +%Y%m%d).tar.gz config/ deployment/
   ```

### Disaster Recovery

1. **Automated Backups**:
   - Schedule regular backups
   - Store backups in multiple locations
   - Test restore procedures

2. **High Availability**:
   - Multi-region deployment
   - Database replication
   - Load balancer failover

## Maintenance

### Regular Tasks

1. **Updates**:
   ```bash
   # Update Docker images
   docker-compose pull
   docker-compose up -d
   
   # Update Kubernetes deployments
   kubectl set image deployment/rag-engine rag-engine=rag-engine:new-tag -n rag-engine
   ```

2. **Cleanup**:
   ```bash
   # Clean up old Docker images
   docker system prune -f
   
   # Clean up old logs
   find /var/log -name "*.log" -mtime +30 -delete
   ```

3. **Health Checks**:
   - Monitor system metrics
   - Check error logs
   - Verify API functionality
   - Test backup procedures

### Upgrade Procedures

1. **Rolling Updates**:
   ```bash
   # Kubernetes rolling update
   kubectl rollout restart deployment/rag-engine -n rag-engine
   kubectl rollout status deployment/rag-engine -n rag-engine
   ```

2. **Blue-Green Deployment**:
   - Deploy new version alongside old
   - Switch traffic gradually
   - Rollback if issues occur

For more detailed information, see the individual component documentation in this directory.