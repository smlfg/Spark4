.PHONY: help build run exec stop clean verify test push pull

# Variables
IMAGE_NAME := spark4-unsloth
IMAGE_TAG := latest
CONTAINER_NAME := spark4-unsloth-trainer
REGISTRY ?= # Set your registry here (e.g., docker.io/username)

# Default target
help:
	@echo "Spark4 Unsloth Training Environment - Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build the Docker image"
	@echo "  make run            - Run container in interactive mode"
	@echo "  make run-bg         - Run container in background"
	@echo "  make exec           - Execute bash in running container"
	@echo "  make verify         - Verify environment inside container"
	@echo "  make stop           - Stop the container"
	@echo "  make clean          - Remove container and image"
	@echo "  make logs           - Show container logs"
	@echo "  make test           - Run environment tests"
	@echo "  make compose-up     - Start services with docker-compose"
	@echo "  make compose-down   - Stop services with docker-compose"
	@echo "  make push           - Push image to registry"
	@echo "  make pull           - Pull image from registry"
	@echo ""

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "Build complete: $(IMAGE_NAME):$(IMAGE_TAG)"

# Build with no cache (clean build)
build-clean:
	@echo "Building Docker image (no cache)..."
	docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "Build complete: $(IMAGE_NAME):$(IMAGE_TAG)"

# Run container in interactive mode
run:
	@echo "Starting container in interactive mode..."
	docker run --gpus all -it --rm \
		--name $(CONTAINER_NAME) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/scripts:/app/scripts \
		-v $(PWD)/outputs:/app/outputs \
		$(IMAGE_NAME):$(IMAGE_TAG)

# Run container in background
run-bg:
	@echo "Starting container in background..."
	docker run --gpus all -d \
		--name $(CONTAINER_NAME) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/scripts:/app/scripts \
		-v $(PWD)/outputs:/app/outputs \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		tail -f /dev/null

# Execute bash in running container
exec:
	@echo "Executing bash in container..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Run verification script
verify:
	@echo "Verifying environment..."
	docker run --gpus all --rm \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python /app/verify_environment.py

# View container logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Stop container
stop:
	@echo "Stopping container..."
	docker stop $(CONTAINER_NAME) || true

# Remove container and image
clean:
	@echo "Cleaning up..."
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm $(CONTAINER_NAME) 2>/dev/null || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@echo "Cleanup complete"

# Run tests
test: verify
	@echo "Running additional tests..."
	docker run --gpus all --rm \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python -c "import torch; import unsloth; print('Quick test passed!')"

# Docker Compose commands
compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d

compose-down:
	@echo "Stopping services..."
	docker-compose down

compose-logs:
	docker-compose logs -f

# Push to registry
push:
	@if [ -z "$(REGISTRY)" ]; then \
		echo "Error: REGISTRY not set. Usage: make push REGISTRY=docker.io/username"; \
		exit 1; \
	fi
	@echo "Tagging image for registry..."
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "Pushing to registry..."
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# Pull from registry
pull:
	@if [ -z "$(REGISTRY)" ]; then \
		echo "Error: REGISTRY not set. Usage: make pull REGISTRY=docker.io/username"; \
		exit 1; \
	fi
	@echo "Pulling from registry..."
	docker pull $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker tag $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG) $(IMAGE_NAME):$(IMAGE_TAG)

# Check GPU availability
check-gpu:
	@echo "Checking GPU availability..."
	nvidia-smi || echo "nvidia-smi not found. Is NVIDIA driver installed?"
	@echo ""
	docker run --gpus all --rm $(IMAGE_NAME):$(IMAGE_TAG) nvidia-smi

# Create required directories
setup-dirs:
	@echo "Creating required directories..."
	mkdir -p data models scripts outputs checkpoints
	@echo "Directories created"
