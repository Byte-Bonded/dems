#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DEMS Monitoring Stack Setup Script
# Grafana + Prometheus for Dynamic Energy Management System
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Check prerequisites
# ─────────────────────────────────────────────────────────────────────────────
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker installed"
    
    # Check Docker Compose (v2)
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
        print_success "Docker Compose v2 available"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        print_success "Docker Compose v1 available"
    else
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        echo "Please start Docker Desktop or the Docker daemon"
        exit 1
    fi
    print_success "Docker daemon running"
}

# ─────────────────────────────────────────────────────────────────────────────
# Start monitoring stack
# ─────────────────────────────────────────────────────────────────────────────
start_stack() {
    print_header "Starting DEMS Monitoring Stack"
    
    # Pull latest images
    echo "Pulling latest images..."
    $COMPOSE_CMD pull
    
    # Start containers
    echo "Starting containers..."
    $COMPOSE_CMD up -d
    
    # Wait for services to be healthy
    echo "Waiting for services to be ready..."
    sleep 5
    
    # Check Prometheus
    if curl -s "http://localhost:9090/-/ready" > /dev/null 2>&1; then
        print_success "Prometheus is ready"
    else
        print_warning "Prometheus may still be starting..."
    fi
    
    # Check Grafana
    if curl -s "http://localhost:3000/api/health" > /dev/null 2>&1; then
        print_success "Grafana is ready"
    else
        print_warning "Grafana may still be starting..."
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Stop monitoring stack
# ─────────────────────────────────────────────────────────────────────────────
stop_stack() {
    print_header "Stopping DEMS Monitoring Stack"
    $COMPOSE_CMD down
    print_success "Monitoring stack stopped"
}

# ─────────────────────────────────────────────────────────────────────────────
# Show logs
# ─────────────────────────────────────────────────────────────────────────────
show_logs() {
    $COMPOSE_CMD logs -f "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────
show_status() {
    print_header "DEMS Monitoring Stack Status"
    $COMPOSE_CMD ps
    
    echo ""
    echo "Service URLs:"
    echo "  Grafana:    http://localhost:3000 (admin/dems2024)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Clean up volumes
# ─────────────────────────────────────────────────────────────────────────────
cleanup() {
    print_header "Cleaning Up"
    
    read -p "This will delete all monitoring data. Are you sure? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $COMPOSE_CMD down -v
        print_success "Volumes removed"
    else
        print_warning "Cleanup cancelled"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Print usage
# ─────────────────────────────────────────────────────────────────────────────
print_usage() {
    echo "DEMS Monitoring Stack Management"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start     Start the monitoring stack (Grafana + Prometheus)"
    echo "  stop      Stop the monitoring stack"
    echo "  restart   Restart the monitoring stack"
    echo "  status    Show status of containers"
    echo "  logs      Show container logs (use -f for follow)"
    echo "  cleanup   Remove all containers and volumes"
    echo "  help      Show this help message"
    echo ""
    echo "Quick Start:"
    echo "  $0 start"
    echo ""
    echo "Access URLs:"
    echo "  Grafana:    http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    echo "Default Grafana Login:"
    echo "  Username: admin"
    echo "  Password: dems2024"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    start)
        check_prerequisites
        start_stack
        show_status
        ;;
    stop)
        check_prerequisites
        stop_stack
        ;;
    restart)
        check_prerequisites
        stop_stack
        start_stack
        show_status
        ;;
    status)
        check_prerequisites
        show_status
        ;;
    logs)
        check_prerequisites
        shift
        show_logs "$@"
        ;;
    cleanup)
        check_prerequisites
        cleanup
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        print_error "Unknown command: $1"
        print_usage
        exit 1
        ;;
esac
