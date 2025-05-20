# Who's My Good Boy?

## Project Overview

"Who's My Good Boy?" is a machine learning portfolio project designed to
showcase skills in deep learning, model deployment, and full-stack development.
The application uses computer vision to classify images in three progressive
stages:

1. **Dog Detection** - Identifying dogs in images using a fine-tuned MobileNet
   V2 model (implemented as a cat-dog-other classifier)
2. **Apolo Recognition** - Further fine-tuning to recognize Apolo (the
   developer's dog) in images
3. **Image Classification** - General image classification using pre-trained
   models

## Repository Structure

This repository serves as an index repository that contains submodules linking
to the individual components of the project:

- `backend/`: FastAPI service for model serving
  ([wmgb-backend](https://github.com/aarangop/wmgb-backend)) - **Available**
- `infrastructure/`: Terraform configurations for AWS deployment
  ([wmgb-infrastructure](https://github.com/aarangop/wmgb-infrastructure.git)) -
  **Available**
- `frontend/`: Next.js web application for user interaction - **Not started
  yet** (repository will be created in the future)
- Machine learning models, training scripts, and evaluation tools currently on
  Kaggle:
  - [Cat-Dog-Other Classifier Fine Tuning](https://www.kaggle.com/code/aarangop/cat-dog-other-model-fine-tuning)
  - [Cat-Dog-Other Model](https://www.kaggle.com/models/aarangop/cat-dog-other-classifier)
  - [Cat-Dog-Other Dataset](https://www.kaggle.com/datasets/aarangop/cat-dog-other)

> **Note:** This project is being actively developed. Currently, only the
> backend and infrastructure components have dedicated repositories. As
> development progresses, separate repositories for frontend and ML components
> will be created and added as submodules.

## Technical Architecture

### Backend

- **FastAPI** for the API server
- **TensorFlow/Keras** for ML model serving
- **Docker** for containerization

### Frontend

- **Next.js** with TypeScript for the user interface
- Image upload and camera capture functionality

### Infrastructure & Deployment

- **Terraform** for infrastructure as code
- **S3** for model storage and versioning
- **AWS ECS and Fargate** for containerized deployment

## Development Status

### Completed

- Project structure and repository setup
- FastAPI backend scaffolding with endpoints
- Docker configuration for containerized development and deployment
- Initial model exploration using pre-trained MobileNetV2
- Infrastructure setup with Terraform
- Migration from monorepo to index repository with submodules

### In Progress

- Fine-tuning models for dog detection and Apolo recognition
- Frontend development planning
- CI/CD pipeline setup for backend service

### Next Steps

1. Evaluate the base dog detector on test datasets
2. Fine-tune the model on a dog/not-dog dataset
3. Complete the frontend implementation
4. Enhance deployment infrastructure and model versioning

## Getting Started

### Clone the Repository with Submodules

```bash
# Clone the main repository
git clone https://github.com/aarangop/whos-my-good-boy.git

# Initialize and update all submodules
cd whos-my-good-boy
git submodule update --init --recursive
```

### Working with Submodules

Each submodule has its own README with specific instructions for setup and
development.
