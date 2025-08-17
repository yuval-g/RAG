#!/bin/bash

# RAG Engine Deployment Script
set -e

# Configuration
ENVIRONMENT=${1:-development}
NAMESPACE="rag-engine"
IMAGE_TAG=${2:-latest}

echo "🚀 Deploying RAG Engine to $ENVIRONMENT environment..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    if ! command_exists podman; then
        echo "❌ Podman is not installed"
        exit 1
    fi
    
    if ! command_exists "podman compose"; then
        echo "❌ Podman Compose is not installed"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Build Docker image
build_image() {
    echo "🔨 Building Docker image..."
    
    if [ "$ENVIRONMENT" = "development" ]; then
        podman build -f Dockerfile.dev -t rag-engine:$IMAGE_TAG .
    else
        podman build -f Dockerfile -t rag-engine:$IMAGE_TAG .
    fi
    
    echo "✅ Docker image built successfully"
}

# Deploy with Podman Compose
deploy_compose() {
    echo "🚢 Deploying with Podman Compose..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "📝 Creating .env file from template..."
        cp .env.example .env
        echo "⚠️  Please update .env file with your configuration"
    fi
    
    if [ "$ENVIRONMENT" = "production" ]; then
        podman compose -f docker-compose.prod.yml up -d
    else
        podman compose up -d
    fi
    
    echo "✅ Services deployed successfully"
}

# Deploy to Kubernetes
deploy_k8s() {
    echo "☸️  Deploying to Kubernetes..."
    
    if ! command_exists kubectl; then
        echo "❌ kubectl is not installed"
        exit 1
    fi
    
    # Create namespace
    kubectl apply -f deployment/k8s/namespace.yaml
    
    # Apply configurations
    kubectl apply -f deployment/k8s/configmap.yaml
    kubectl apply -f deployment/k8s/secrets.yaml
    
    # Deploy services
    kubectl apply -f deployment/k8s/redis-deployment.yaml
    kubectl apply -f deployment/k8s/chroma-deployment.yaml
    kubectl apply -f deployment/k8s/rag-engine-deployment.yaml
    kubectl apply -f deployment/k8s/nginx-deployment.yaml
    kubectl apply -f deployment/k8s/monitoring.yaml
    
    # Wait for deployments
    echo "⏳ Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/chroma -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/rag-engine -n $NAMESPACE
    
    echo "✅ Kubernetes deployment completed"
}

# Health check
health_check() {
    echo "🏥 Performing health check..."
    
    if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        # Get service URL for K8s
        SERVICE_URL=$(kubectl get service nginx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -z "$SERVICE_URL" ]; then
            SERVICE_URL="localhost"
        fi
    else
        SERVICE_URL="localhost"
    fi
    
    # Wait for service to be ready
    echo "⏳ Waiting for service to be ready..."
    for i in {1..30}; do
        if curl -f http://$SERVICE_URL/health >/dev/null 2>&1; then
            echo "✅ Service is healthy"
            break
        fi
        echo "⏳ Attempt $i/30: Service not ready yet..."
        sleep 10
    done
    
    if [ $i -eq 30 ]; then
        echo "❌ Service health check failed"
        exit 1
    fi
}

# Show deployment info
show_info() {
    echo "📊 Deployment Information:"
    echo "Environment: $ENVIRONMENT"
    echo "Image Tag: $IMAGE_TAG"
    echo "Namespace: $NAMESPACE"
    
    if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        echo ""
        echo "Kubernetes Resources:"
        kubectl get all -n $NAMESPACE
        
        echo ""
        echo "Service URLs:"
        kubectl get ingress -n $NAMESPACE
    else
        echo ""
        echo "Docker Compose Services:"
        podman compose ps
        
        echo ""
        echo "Service URLs:"
        echo "API: http://localhost:8000"
        echo "Health: http://localhost:8089"
        echo "Grafana: http://localhost:3000"
        echo "Prometheus: http://localhost:9090"
    fi
}

# Main deployment logic
main() {
    check_prerequisites
    
    # Determine deployment type
    if [ "$ENVIRONMENT" = "k8s" ] || [ "$ENVIRONMENT" = "kubernetes" ]; then
        DEPLOYMENT_TYPE="k8s"
        ENVIRONMENT="production"
    else
        DEPLOYMENT_TYPE="compose"
    fi
    
    build_image
    
    if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        deploy_k8s
    else
        deploy_compose
    fi
    
    health_check
    show_info
    
    echo ""
    echo "🎉 RAG Engine deployment completed successfully!"
    echo "📚 Check the documentation in deployment/docs/ for more information"
}

# Run main function
main "$@"