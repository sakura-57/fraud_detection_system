.PHONY: help install test lint format clean run docker-build docker-run

# Colors for output
GREEN := \033[0;32m
NC := \033[0m # No Color

# Default target
help:
	@echo "Available commands:"
	@echo "  ${GREEN}install${NC}     - Install dependencies"
	@echo "  ${GREEN}test${NC}        - Run tests"
	@echo "  ${GREEN}lint${NC}        - Check code style"
	@echo "  ${GREEN}format${NC}      - Format code"
	@echo "  ${GREEN}run${NC}         - Run API locally"
	@echo "  ${GREEN}docker-build${NC} - Build Docker image"
	@echo "  ${GREEN}docker-run${NC}  - Run with Docker Compose"
	@echo "  ${GREEN}clean${NC}       - Clean temporary files"

# Installation
install:
	@echo "${GREEN}Installing dependencies...${NC}"
	pip install -r requirements.txt
	@echo "${GREEN}✓ Dependencies installed${NC}"

# Testing
test:
	@echo "${GREEN}Running tests...${NC}"
	pytest tests/ -v

# Code quality
lint:
	@echo "${GREEN}Checking code style...${NC}"
	flake8 src/ tests/

format:
	@echo "${GREEN}Formatting code...${NC}"
	black src/ tests/

# Development server
run:
	@echo "${GREEN}Starting development server...${NC}"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Docker commands
docker-build:
	@echo "${GREEN}Building Docker image...${NC}"
	docker build -t fraud-detection-api .

docker-run:
	@echo "${GREEN}Starting with Docker Compose...${NC}"
	docker-compose up

# Cleanup
clean:
	@echo "${GREEN}Cleaning temporary files...${NC}"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "${GREEN}✓ Cleanup complete${NC}"

# Setup (first time)
setup: install
	@echo "${GREEN}Creating .env file from template...${NC}"
	cp .env.example .env
	@echo "${GREEN}✓ Please edit .env file with your configuration${NC}"
	@echo "${GREEN}Creating log directory...${NC}"
	mkdir -p logs
	@echo "${GREEN}✓ Setup complete${NC}"