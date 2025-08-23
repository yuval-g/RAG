# Maintenance Guide

This guide outlines the routine maintenance tasks and best practices for ensuring the smooth and reliable operation of the RAG System in production.

## Overview

Regular maintenance is crucial for the long-term health, performance, and security of any deployed system. This includes keeping software up-to-date, cleaning up old data, and verifying system health.

## Table of Contents

1.  [Regular Maintenance Tasks](#regular-maintenance-tasks)
2.  [Automated Maintenance](#automated-maintenance)
3.  [Backup and Recovery](#backup-and-recovery)
4.  [Updates and Upgrades](#updates-and-upgrades)
5.  [Health Checks and Monitoring](#health-checks-and-monitoring)

## 1. Regular Maintenance Tasks

These tasks should be performed periodically to keep the RAG System running optimally.

### 1.1. Log Management

Logs can consume significant disk space over time. Implement a log rotation strategy.

*   **Cleanup Old Logs**: Regularly remove log files older than a certain period (e.g., 7 or 30 days).
```bash
# Example: Delete logs older than 7 days
find /path/to/rag-engine/logs -name "*.log" -mtime +7 -delete

```
*   **Configure Log Rotation**: For Docker, configure log drivers. For bare-metal, use `logrotate`.
```yaml
# Example Docker Compose logging configuration
logging:
  driver: "json-file"
  options:
    max-size: "10m" # Max size of the log file before rotation
    max-file: "3"   # Max number of log files to keep

```

### 1.2. Temporary File Cleanup

Ensure temporary directories are regularly cleared to prevent disk space issues.

*   **Cleanup `/tmp` and `/var/tmp`**: If your application uses these directories, ensure they are managed.
```bash
# Example: Clean up temporary files older than 1 day
find /tmp -type f -mtime +1 -delete

```

### 1.3. Docker/Container Cleanup (if applicable)

If deploying with Docker, regularly prune unused Docker objects.

*   **Prune System**: Remove stopped containers, unused networks, dangling images, and build cache.
```bash
docker system prune -f

```
*   **Prune Volumes**: Remove unused local volumes (use with caution, ensure volumes are truly unused).
```bash
docker volume prune -f

```

### 1.4. Database Maintenance (ChromaDB, Redis)

*   **ChromaDB**: If using a persistent ChromaDB, ensure its data directory is regularly backed up. Depending on usage, occasional re-indexing or optimization might be beneficial.
*   **Redis**: Ensure Redis persistence (AOF or RDB) is configured and backups are taken.

## 2. Automated Maintenance

Automate routine tasks using cron jobs (for bare-metal/VMs) or Kubernetes CronJobs (for Kubernetes deployments).

### Example Automated Maintenance Script

Create a script (e.g., `maintenance.sh`):

```bash
#!/bin/bash
# maintenance.sh

echo "Starting RAG Engine maintenance: $(date)"

# Update and restart services (if using Docker Compose)
# docker-compose pull
# docker-compose up -d

# Clean up old Docker images (e.g., older than 7 days)
# docker image prune -f --filter "until=168h"

# Clean up application logs
find /path/to/rag-engine/logs -name "*.log" -mtime +7 -delete

# Clean up temporary files
find /tmp -type f -mtime +1 -delete

# Run health checks (if applicable)
# ./deployment/scripts/health-check.sh

# Run backup (if applicable)
# ./deployment/scripts/backup.sh

echo "RAG Engine maintenance completed: $(date)"
```

### Scheduling with Cron (Linux/macOS)

Add an entry to your crontab to run the script periodically (e.g., weekly at 2 AM):

```bash
# Open crontab editor
crontab -e

# Add the following line
0 2 * * 0 /path/to/your/maintenance.sh >> /var/log/rag-engine-maintenance.log 2>&1
```

### Scheduling with Kubernetes CronJob

For Kubernetes deployments, use a `CronJob` resource:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rag-engine-cleanup
  namespace: rag-engine
spec:
  schedule: "0 2 * * 0" # Weekly at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: alpine:latest # Use a minimal image with necessary tools
            command:
            - /bin/sh
            - -c
            - |
              # Example: Clean up logs in a mounted volume
              find /app/logs -name "*.log" -mtime +7 -delete
              # Example: Clean up temporary files
              find /tmp -type f -mtime +1 -delete
            volumeMounts:
            - name: logs-volume # Mount the volume where logs are stored
              mountPath: /app/logs
          volumes:
          - name: logs-volume
            persistentVolumeClaim:
              claimName: rag-engine-logs-pvc # Replace with your actual PVC
          restartPolicy: OnFailure
```

## 3. Backup and Recovery

Regularly back up your critical data and ensure you have a tested recovery plan.

*   **Data to Backup**: This typically includes:
    *   ChromaDB persistent data (if using local persistence).
    *   Redis data (if using Redis for persistent caching).
    *   Application configuration files.
    *   Any custom data or models used by the RAG Engine.
*   **Backup Frequency**: Determine based on your data change rate and recovery point objective (RPO).
*   **Storage Location**: Store backups securely, preferably off-site or in a different region.
*   **Test Recovery**: Periodically test your recovery procedures to ensure they work as expected.

**Relevant Documentation:**
*   [Docker Deployment Guide](../deployment/docker.md) (includes backup/restore scripts)
*   [Kubernetes Deployment Guide](../deployment/kubernetes.md) (includes Velero backup examples)

## 4. Updates and Upgrades

Keep your RAG System and its dependencies up-to-date to benefit from new features, performance improvements, and security patches.

*   **Application Updates**: Regularly pull the latest changes from the RAG System repository and redeploy.
*   **Dependency Updates**: Update Python packages (`uv sync` or `pip install --upgrade -r requirements.txt`), Docker base images, and other external services.
*   **Deployment Strategy**: Use rolling updates (Kubernetes) or recreate containers (Docker Compose) to minimize downtime during updates.

**Relevant Documentation:**
*   [Docker Deployment Guide](../deployment/docker.md)
*   [Kubernetes Deployment Guide](../deployment/kubernetes.md)

## 5. Health Checks and Monitoring

Continuous monitoring is a form of proactive maintenance. Ensure your monitoring systems are configured correctly.

*   **Verify Health Endpoints**: Regularly check the `/health` and `/status` endpoints of your RAG API.
*   **Monitor Metrics**: Keep an eye on key performance indicators (KPIs) like response times, error rates, CPU/memory usage, and indexed document counts.
*   **Alerting**: Set up alerts for any deviations from normal operating parameters.

**Relevant Documentation:**
*   [Monitoring and Health Checks](./monitoring.md)
*   [Troubleshooting Guide](./troubleshooting.md)
