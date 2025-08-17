#!/bin/bash

# RAG Engine Cleanup Script
set -e

ENVIRONMENT=${1:-development}
NAMESPACE="rag-engine"

echo "🧹 Cleaning up RAG Engine deployment..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Cleanup Docker Compose
cleanup_compose() {
    echo "🐳 Cleaning up Docker Compose deployment..."
    
    if [ -f docker-compose.yml ]; then
        if [ "$ENVIRONMENT" = "production" ]; then
            podman compose -f docker-compose.prod.yml down -v
        else
            podman compose down -v
        fi
        
        echo "🗑️  Removing unused Docker resources..."
        podman system prune -f
        podman volume prune -f
        
        echo "✅ Docker Compose cleanup completed"
    else
        echo "⚠️  No docker-compose.yml found"
    fi
}

# Cleanup Kubernetes
cleanup_k8s() {
    echo "☸️  Cleaning up Kubernetes deployment..."
    
    if ! command_exists kubectl; then
        echo "❌ kubectl is not installed"
        exit 1
    fi
    
    # Check if namespace exists
    if kubectl get namespace $NAMESPACE >/dev/null 2>&1; then
        echo "🗑️  Deleting Kubernetes resources..."
        
        # Delete all resources in namespace
        kubectl delete all --all -n $NAMESPACE
        kubectl delete pvc --all -n $NAMESPACE
        kubectl delete configmap --all -n $NAMESPACE
        kubectl delete secret --all -n $NAMESPACE
        kubectl delete ingress --all -n $NAMESPACE
        
        # Delete namespace
        kubectl delete namespace $NAMESPACE
        
        echo "✅ Kubernetes cleanup completed"
    else
        echo "⚠️  Namespace $NAMESPACE not found"
    fi
}

# Remove Docker images
cleanup_images() {
    echo "🖼️  Cleaning up Docker images..."
    
    # Remove RAG Engine images
    podman images | grep rag-engine | awk '{print $3}' | xargs -r docker rmi -f
    
    echo "✅ Docker images cleanup completed"
}

# Main cleanup logic
main() {
    echo "Environment: $ENVIRONMENT"
    
    # Determine deployment type
    if [ "$ENVIRONMENT" = "k8s" ] || [ "$ENVIRONMENT" = "kubernetes" ]; then
        cleanup_k8s
    else
        cleanup_compose
    fi
    
    # Ask if user wants to remove images
    read -p "Do you want to remove Docker images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_images
    fi
    
    echo ""
    echo "🎉 Cleanup completed successfully!"
}

# Run main function
main "$@"