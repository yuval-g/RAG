# Kubernetes Deployment Guide

This guide covers deploying the RAG Engine on Kubernetes for production environments.

## Prerequisites

### Cluster Requirements

- **Kubernetes Version**: 1.24+
- **Node Resources**: Minimum 3 nodes with 4 CPU, 8GB RAM each
- **Storage**: Dynamic provisioning with SSD storage classes
- **Networking**: CNI plugin (Calico, Flannel, or Weave)
- **Ingress**: Nginx Ingress Controller or similar

### Required Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm (optional)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

## Quick Deployment

### Automated Deployment

```bash
# Clone repository
git clone <repository-url>
cd rag-engine

# Deploy to Kubernetes
./deployment/scripts/deploy.sh kubernetes

# Check deployment status
kubectl get all -n rag-engine
```

### Manual Deployment

```bash
# Create namespace
kubectl apply -f deployment/k8s/namespace.yaml

# Create secrets (update with your values)
kubectl create secret generic rag-engine-secrets \
  --from-literal=GOOGLE_API_KEY=your_google_api_key \
  --from-literal=GRAFANA_PASSWORD=your_grafana_password \
  -n rag-engine

# Deploy services
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/redis-deployment.yaml
kubectl apply -f deployment/k8s/chroma-deployment.yaml
kubectl apply -f deployment/k8s/rag-engine-deployment.yaml
kubectl apply -f deployment/k8s/nginx-deployment.yaml
kubectl apply -f deployment/k8s/monitoring.yaml

# Wait for deployments
kubectl wait --for=condition=available --timeout=300s deployment --all -n rag-engine
```

## Architecture Overview

### Service Topology

```
Internet
    ↓
[Load Balancer]
    ↓
[Nginx Ingress]
    ↓
[RAG Engine Pods] ←→ [Chroma DB] ←→ [Redis Cache]
    ↓
[Monitoring Stack]
```

### Resource Distribution

| Component | Replicas | CPU Request | Memory Request | Storage |
|-----------|----------|-------------|----------------|---------|
| RAG Engine | 3 | 500m | 1Gi | - |
| Chroma | 1 | 250m | 512Mi | 20Gi |
| Redis | 1 | 100m | 256Mi | 5Gi |
| Nginx | 2 | 100m | 128Mi | - |
| Prometheus | 1 | 500m | 1Gi | 50Gi |
| Grafana | 1 | 100m | 256Mi | 5Gi |

## Configuration Management

### Secrets Management

#### Create Secrets

```bash
# API Keys
kubectl create secret generic rag-engine-secrets \
  --from-literal=GOOGLE_API_KEY="your_google_api_key" \
  --from-literal=COHERE_API_KEY="your_cohere_api_key" \
  --from-literal=GRAFANA_PASSWORD="secure_password" \
  -n rag-engine

# SSL Certificates
kubectl create secret tls ssl-certs \
  --cert=path/to/your/cert.pem \
  --key=path/to/your/key.pem \
  -n rag-engine

# Verify secrets
kubectl get secrets -n rag-engine
```

#### External Secrets (Recommended)

For production, use external secret management:

```yaml
# external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: rag-engine
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "rag-engine"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: rag-engine-secrets
  namespace: rag-engine
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: rag-engine-secrets
    creationPolicy: Owner
  data:
  - secretKey: GOOGLE_API_KEY
    remoteRef:
      key: rag-engine
      property: google_api_key
```

### ConfigMaps

```yaml
# Enhanced ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-engine-config
  namespace: rag-engine
data:
  # Application Configuration
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  WORKERS: "4"
  
  # Service Discovery
  CHROMA_HOST: "chroma-service"
  CHROMA_PORT: "8000"
  REDIS_URL: "redis://redis-service:6379"
  
  # Performance Tuning
  MAX_CONCURRENT_REQUESTS: "100"
  CACHE_TTL: "3600"
  CHUNK_SIZE: "1000"
  RETRIEVAL_K: "5"
  
  # Feature Flags
  ENABLE_RERANKING: "true"
  ENABLE_MONITORING: "true"
  ENABLE_CACHING: "true"
```

## Deployment Configurations

### RAG Engine Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine
  namespace: rag-engine
  labels:
    app: rag-engine
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: rag-engine
  template:
    metadata:
      labels:
        app: rag-engine
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8089"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: rag-engine
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: rag-engine
        image: rag-engine:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: api
          protocol: TCP
        - containerPort: 8089
          name: metrics
          protocol: TCP
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-engine-secrets
              key: GOOGLE_API_KEY
        envFrom:
        - configMapRef:
            name: rag-engine-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
            ephemeral-storage: "1Gi"
          limits:
            memory: "2Gi"
            cpu: "1000m"
            ephemeral-storage: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8089
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8089
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        startupProbe:
          httpGet:
            path: /health
            port: 8089
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
        - name: tmp-volume
          mountPath: /tmp
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: rag-engine-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: rag-engine-logs-pvc
      - name: tmp-volume
        emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - rag-engine
              topologyKey: kubernetes.io/hostname
      tolerations:
      - key: "rag-engine"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### Service Account and RBAC

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rag-engine
  namespace: rag-engine

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: rag-engine
  name: rag-engine-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: rag-engine-rolebinding
  namespace: rag-engine
subjects:
- kind: ServiceAccount
  name: rag-engine
  namespace: rag-engine
roleRef:
  kind: Role
  name: rag-engine-role
  apiGroup: rbac.authorization.k8s.io
```

## Storage Configuration

### Storage Classes

```yaml
# Fast SSD Storage Class
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # Adjust for your cloud provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true

---
# Standard Storage Class
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

### Persistent Volume Claims

```yaml
# RAG Engine Data PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rag-engine-data-pvc
  namespace: rag-engine
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd

---
# Chroma Database PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chroma-data-pvc
  namespace: rag-engine
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd

---
# Prometheus Data PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-data-pvc
  namespace: rag-engine
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

## Networking and Ingress

### Service Configuration

```yaml
# RAG Engine Service
apiVersion: v1
kind: Service
metadata:
  name: rag-engine-service
  namespace: rag-engine
  labels:
    app: rag-engine
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8089"
    prometheus.io/path: "/metrics"
spec:
  selector:
    app: rag-engine
  ports:
  - name: api
    port: 8000
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 8089
    targetPort: 8089
    protocol: TCP
  type: ClusterIP
  sessionAffinity: None
```

### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-engine-ingress
  namespace: rag-engine
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    - monitoring.yourdomain.com
    secretName: rag-engine-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: rag-engine-service
            port:
              number: 8000
      - path: /health
        pathType: Prefix
        backend:
          service:
            name: rag-engine-service
            port:
              number: 8089
  - host: monitoring.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana-service
            port:
              number: 3000
```

## Auto-scaling

### Horizontal Pod Autoscaler

```yaml
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
  maxReplicas: 20
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
  - type: Pods
    pods:
      metric:
        name: rag_engine_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: rag-engine-vpa
  namespace: rag-engine
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-engine
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: rag-engine
      minAllowed:
        cpu: 100m
        memory: 512Mi
      maxAllowed:
        cpu: 2000m
        memory: 4Gi
      controlledResources: ["cpu", "memory"]
```

### Cluster Autoscaler

```yaml
# Node pool configuration (example for AWS EKS)
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  nodes.max: "20"
  nodes.min: "3"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: rag-engine
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'rag-engine-k8s'
        environment: 'production'

    rule_files:
      - "/etc/prometheus/rules/*.yml"

    scrape_configs:
      # RAG Engine metrics
      - job_name: 'rag-engine'
        kubernetes_sd_configs:
        - role: endpoints
          namespaces:
            names:
            - rag-engine
        relabel_configs:
        - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_service_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\d+)?;(\d+)
          replacement: $1:$2
          target_label: __address__

      # Kubernetes cluster metrics
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
        - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
          action: keep
          regex: default;kubernetes;https

      # Node metrics
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
        - role: node
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - action: labelmap
          regex: __meta_kubernetes_node_label_(.+)

    alerting:
      alertmanagers:
      - kubernetes_sd_configs:
        - role: pod
          namespaces:
            names:
            - rag-engine
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: keep
          regex: alertmanager
```

### Alert Rules

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: rag-engine
data:
  rag-engine.yml: |
    groups:
    - name: rag-engine
      rules:
      - alert: RAGEngineDown
        expr: up{job="rag-engine"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "RAG Engine instance is down"
          description: "RAG Engine instance {{ $labels.instance }} has been down for more than 1 minute."

      - alert: RAGEngineHighErrorRate
        expr: rate(rag_engine_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in RAG Engine"
          description: "RAG Engine error rate is {{ $value }} errors per second."

      - alert: RAGEngineHighLatency
        expr: histogram_quantile(0.95, rate(rag_engine_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency in RAG Engine"
          description: "RAG Engine 95th percentile latency is {{ $value }} seconds."

      - alert: RAGEngineHighMemoryUsage
        expr: container_memory_usage_bytes{pod=~"rag-engine-.*"} / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage in RAG Engine"
          description: "RAG Engine pod {{ $labels.pod }} memory usage is above 90%."

      - alert: ChromaDBDown
        expr: up{job="chroma"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Chroma database is down"
          description: "Chroma database has been down for more than 1 minute."

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis cache is down"
          description: "Redis cache has been down for more than 1 minute."
```

## Security

### Pod Security Standards

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-engine
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-engine-network-policy
  namespace: rag-engine
spec:
  podSelector:
    matchLabels:
      app: rag-engine
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: nginx
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8089
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: chroma
    ports:
    - protocol: TCP
      port: 8000
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []  # Allow external API calls
    ports:
    - protocol: TCP
      port: 443
```

### Security Context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
    - ALL
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
```

## Backup and Disaster Recovery

### Backup Strategy

```yaml
# Velero backup configuration
apiVersion: velero.io/v1
kind: Backup
metadata:
  name: rag-engine-backup
  namespace: velero
spec:
  includedNamespaces:
  - rag-engine
  storageLocation: default
  ttl: 720h0m0s  # 30 days
  includedResources:
  - persistentvolumes
  - persistentvolumeclaims
  - secrets
  - configmaps
  hooks:
    resources:
    - name: chroma-backup-hook
      includedNamespaces:
      - rag-engine
      labelSelector:
        matchLabels:
          app: chroma
      pre:
      - exec:
          container: chroma
          command:
          - /bin/sh
          - -c
          - "tar -czf /tmp/chroma-backup.tar.gz /chroma/chroma"
      post:
      - exec:
          container: chroma
          command:
          - /bin/sh
          - -c
          - "rm -f /tmp/chroma-backup.tar.gz"
```

### Disaster Recovery Plan

1. **Automated Backups**:
   ```bash
   # Schedule daily backups
   kubectl create cronjob rag-engine-backup \
     --image=velero/velero:latest \
     --schedule="0 2 * * *" \
     --restart=OnFailure \
     -- velero backup create rag-engine-$(date +%Y%m%d)
   ```

2. **Cross-Region Replication**:
   ```yaml
   # Configure cross-region storage
   apiVersion: v1
   kind: Secret
   metadata:
     name: cloud-credentials
     namespace: velero
   data:
     cloud: <base64-encoded-credentials>
   ```

3. **Recovery Procedures**:
   ```bash
   # Restore from backup
   velero restore create --from-backup rag-engine-20240101
   
   # Verify restoration
   kubectl get all -n rag-engine
   ```

## Troubleshooting

### Common Issues

1. **Pod Startup Issues**:
   ```bash
   # Check pod status
   kubectl get pods -n rag-engine
   kubectl describe pod <pod-name> -n rag-engine
   kubectl logs <pod-name> -n rag-engine
   
   # Check events
   kubectl get events -n rag-engine --sort-by='.lastTimestamp'
   ```

2. **Resource Issues**:
   ```bash
   # Check resource usage
   kubectl top pods -n rag-engine
   kubectl top nodes
   
   # Check resource quotas
   kubectl describe resourcequota -n rag-engine
   ```

3. **Network Issues**:
   ```bash
   # Test connectivity
   kubectl exec -it <pod-name> -n rag-engine -- curl http://chroma-service:8000/api/v1/heartbeat
   
   # Check DNS resolution
   kubectl exec -it <pod-name> -n rag-engine -- nslookup chroma-service
   ```

4. **Storage Issues**:
   ```bash
   # Check PVC status
   kubectl get pvc -n rag-engine
   kubectl describe pvc <pvc-name> -n rag-engine
   
   # Check storage class
   kubectl get storageclass
   ```

### Debug Commands

```bash
# Get comprehensive cluster info
kubectl cluster-info dump --namespaces rag-engine --output-directory=./cluster-dump

# Check resource consumption
kubectl describe nodes
kubectl get pods -o wide -n rag-engine

# Network debugging
kubectl run debug --image=nicolaka/netshoot -it --rm -- /bin/bash

# Storage debugging
kubectl get pv,pvc -n rag-engine
kubectl describe pv <pv-name>
```

### Performance Tuning

1. **Node Optimization**:
   ```bash
   # Label nodes for specific workloads
   kubectl label nodes <node-name> workload=rag-engine
   
   # Taint nodes for dedicated use
   kubectl taint nodes <node-name> rag-engine=true:NoSchedule
   ```

2. **Resource Optimization**:
   ```yaml
   # Use resource quotas
   apiVersion: v1
   kind: ResourceQuota
   metadata:
     name: rag-engine-quota
     namespace: rag-engine
   spec:
     hard:
       requests.cpu: "10"
       requests.memory: 20Gi
       limits.cpu: "20"
       limits.memory: 40Gi
       persistentvolumeclaims: "10"
   ```

3. **Scheduling Optimization**:
   ```yaml
   # Use node affinity
   affinity:
     nodeAffinity:
       requiredDuringSchedulingIgnoredDuringExecution:
         nodeSelectorTerms:
         - matchExpressions:
           - key: kubernetes.io/instance-type
             operator: In
             values:
             - c5.2xlarge
             - c5.4xlarge
   ```

## Maintenance

### Rolling Updates

```bash
# Update deployment image
kubectl set image deployment/rag-engine rag-engine=rag-engine:v2.0.0 -n rag-engine

# Monitor rollout
kubectl rollout status deployment/rag-engine -n rag-engine

# Rollback if needed
kubectl rollout undo deployment/rag-engine -n rag-engine
```

### Cluster Maintenance

```bash
# Drain node for maintenance
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Cordon node (prevent new pods)
kubectl cordon <node-name>

# Uncordon node (allow new pods)
kubectl uncordon <node-name>
```

### Automated Maintenance

```yaml
# CronJob for cleanup
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-job
  namespace: rag-engine
spec:
  schedule: "0 2 * * 0"  # Weekly at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: alpine:latest
            command:
            - /bin/sh
            - -c
            - |
              # Cleanup old logs
              find /app/logs -name "*.log" -mtime +7 -delete
              # Cleanup temporary files
              find /tmp -type f -mtime +1 -delete
            volumeMounts:
            - name: logs-volume
              mountPath: /app/logs
          volumes:
          - name: logs-volume
            persistentVolumeClaim:
              claimName: rag-engine-logs-pvc
          restartPolicy: OnFailure
```

This comprehensive Kubernetes guide covers all aspects of deploying and managing the RAG Engine in a production Kubernetes environment.