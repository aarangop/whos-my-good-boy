#!/bin/bash

# Create main directory structure
mkdir -p .github/workflows
mkdir -p backend/app
mkdir -p backend/models
mkdir -p frontend/src
mkdir -p frontend/public
mkdir -p ml/notebooks
mkdir -p ml/training
mkdir -p ml/evaluation
mkdir -p ml/datasets
mkdir -p infrastructure/terraform
mkdir -p docs

# Create initial files
touch .github/workflows/ci.yml
touch backend/Dockerfile
touch backend/requirements.txt
touch frontend/package.json
touch infrastructure/terraform/main.tf
touch README.md

# Add minimal content to README
cat > README.md << 'EOF'
# Who's My Good Boy?

An AI-powered web application that uses deep learning to:
1. Classify images using a pre-trained model
2. Identify dogs in images using a fine-tuned model
3. Recognize Apolo (my dog) in images

## Project Structure
- `backend/`: FastAPI service for model serving
- `frontend/`: Next.js web application
- `ml/
