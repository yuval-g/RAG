#!/bin/bash

# RAG Engine Health Check Script
set -e

# Configuration
ENVIRONMENT=${1:-development}
NAMESPACE="rag-engine"
TIMEOUT=30

echo "🏥 Running health checks for RAG Engine ($ENVIRONMENT)..."

# Function to check HTTP endpoint
check_http() {
    local url=$1
    local name=$2
    local timeout=${3:-10}
    
    echo -n "Checking $name... "
    
    if curl -f -s --max-time $timeout "$url" > /dev/null; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

# Function to check Kubernetes deployment
check_k8s_deployment() {
    local deployment=$1
    local namespace=$2
    
    echo -n "Checking K8s deployment $deployment... "
    
    if kubectl get deployment "$deployment" -n "$namespace" &> /dev/null; then
        local ready=$(kubectl get deployment "$deployment" -n "$namespace" -o jsonpath='{.status.readyReplicas}')
        local desired=$(kubectl get deployment "$deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')
        
        if [ "$ready" = "$desired" ] && [ "$ready" -gt 0 ]; then
            echo "✅ OK ($ready/$desired ready)"
            return 0
        else
            echo "❌ FAILED ($ready/$desired ready)"
            return 1
        fi
    else
        echo "❌ NOT FOUND"
        return 1
    fi
}

# Function to check Docker Compose service
check_compose_service() {
    local service=$1
    
    echo -n "Checking Docker Compose service $service... "
    
    if podman compose ps "$service" | grep -q "Up"; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

# Determine deployment type and check accordingly
if kubectl cluster-info &> /dev/null && kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "📋 Detected Kubernetes deployment"
    DEPLOYMENT_TYPE="k8s"
    
    # Check Kubernetes deployments
    check_k8s_deployment "rag-engine" "$NAMESPACE"
    check_k8s_deployment "chroma" "$NAMESPACE"
    check_k8s_deployment "redis" "$NAMESPACE"
    
    # Get service URLs
    RAG_ENGINE_URL="http://$(kubectl get service rag-engine-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):8000"
    HEALTH_URL="http://$(kubectl get service rag-engine-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):8089"
    CHROMA_URL="http://$(kubectl get service chroma-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):8000"
    
elif podman compose ps &> /dev/null; then
    echo "📋 Detected Docker Compose deployment"
    DEPLOYMENT_TYPE="compose"
    
    # Check Docker Compose services
    check_compose_service "rag-engine"
    check_compose_service "chroma"
    check_compose_service "redis"
    
    # Set service URLs
    RAG_ENGINE_URL="http://localhost:8000"
    HEALTH_URL="http://localhost:8089"
    CHROMA_URL="http://localhost:8001"
    
else
    echo "❌ No deployment detected (neither Kubernetes nor Docker Compose)"
    exit 1
fi

echo ""
echo "🌐 Checking service endpoints..."

# Check service health endpoints
FAILED_CHECKS=0

if ! check_http "$HEALTH_URL/health" "RAG Engine Health API"; then
    ((FAILED_CHECKS++))
fi

if ! check_http "$CHROMA_URL/api/v1/heartbeat" "Chroma Database"; then
    ((FAILED_CHECKS++))
fi

# Check API functionality
echo ""
echo "🔧 Testing API functionality..."

# Test basic API endpoint
echo -n "Testing RAG Engine API... "
if curl -f -s --max-time $TIMEOUT "$RAG_ENGINE_URL/health" > /dev/null; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    ((FAILED_CHECKS++))
fi

# Test metrics endpoint
echo -n "Testing metrics endpoint... "
if curl -f -s --max-time $TIMEOUT "$HEALTH_URL/metrics" | grep -q "rag_engine_info"; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    ((FAILED_CHECKS++))
fi

# Additional checks for Kubernetes
if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    echo ""
    echo "☸️  Kubernetes-specific checks..."
    
    # Check pod status
    echo -n "Checking pod status... "
    UNHEALTHY_PODS=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    if [ "$UNHEALTHY_PODS" -eq 0 ]; then
        echo "✅ OK (all pods running)"
    else
        echo "❌ FAILED ($UNHEALTHY_PODS unhealthy pods)"
        ((FAILED_CHECKS++))
    fi
    
    # Check persistent volumes
    echo -n "Checking persistent volumes... "
    UNBOUND_PVCS=$(kubectl get pvc -n "$NAMESPACE" --field-selector=status.phase!=Bound --no-headers 2>/dev/null | wc -l)
    if [ "$UNBOUND_PVCS" -eq 0 ]; then
        echo "✅ OK (all PVCs bound)"
    else
        echo "❌ FAILED ($UNBOUND_PVCS unbound PVCs)"
        ((FAILED_CHECKS++))
    fi
fi

# Additional checks for Docker Compose
if [ "$DEPLOYMENT_TYPE" = "compose" ]; then
    echo ""
    echo "🐳 Docker Compose-specific checks..."
    
    # Check container health
    echo -n "Checking container health... "
    UNHEALTHY_CONTAINERS=$(podman compose ps --filter "health=unhealthy" -q | wc -l)
    if [ "$UNHEALTHY_CONTAINERS" -eq 0 ]; then
        echo "✅ OK (all containers healthy)"
    else
        echo "❌ FAILED ($UNHEALTHY_CONTAINERS unhealthy containers)"
        ((FAILED_CHECKS++))
    fi
    
    # Check volumes
    echo -n "Checking Docker volumes... "
    if podman volume ls | grep -q "rag"; then
        echo "✅ OK (volumes exist)"
    else
        echo "⚠️  WARNING (no RAG volumes found)"
    fi
fi

# Performance checks
echo ""
echo "⚡ Performance checks..."

# Check response time
echo -n "Checking API response time... "
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' --max-time $TIMEOUT "$HEALTH_URL/health" 2>/dev/null || echo "999")
if (( $(echo "$RESPONSE_TIME < 2.0" | bc -l) )); then
    echo "✅ OK (${RESPONSE_TIME}s)"
else
    echo "⚠️  SLOW (${RESPONSE_TIME}s)"
fi

# Check system resources (if available)
if command -v free &> /dev/null; then
    echo -n "Checking system memory... "
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$MEMORY_USAGE" -lt 90 ]; then
        echo "✅ OK (${MEMORY_USAGE}% used)"
    else
        echo "⚠️  HIGH (${MEMORY_USAGE}% used)"
    fi
fi

# Summary
echo ""
echo "📊 Health Check Summary:"
echo "Environment: $ENVIRONMENT"
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "Failed Checks: $FAILED_CHECKS"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo "🎉 All health checks passed!"
    exit 0
else
    echo "❌ $FAILED_CHECKS health check(s) failed!"
    
    # Show troubleshooting tips
    echo ""
    echo "🔍 Troubleshooting tips:"
    if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        echo "- Check pod logs: kubectl logs -l app=rag-engine -n $NAMESPACE"
        echo "- Check events: kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
        echo "- Check pod status: kubectl describe pods -l app=rag-engine -n $NAMESPACE"
    else
        echo "- Check service logs: podman compose logs rag-engine"
        echo "- Check service status: podman compose ps"
        echo "- Restart services: podman compose restart"
    fi
    
    exit 1
fi