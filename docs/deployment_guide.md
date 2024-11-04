# Deployment Guide

This guide explains how to deploy the Phoenix Multimodal LLM in production environments using cloud platforms and optimized binaries.

## 1. Docker Setup

The Docker environment includes both Python and C++ components for seamless deployment.

- **Dockerfile**: Located at `deployment/docker/Dockerfile`.
- **Docker Compose**: Multi-container orchestration is handled via `docker-compose.yml`.

## 2. Cloud Deployment

Scripts for deploying the models across different cloud platforms are provided:

- **AWS**: `deployment/scripts/deploy_aws.py`
- **Google Cloud**: `deployment/scripts/deploy_gcp.py`
- **Azure**: `deployment/scripts/deploy_azure.py`

## 3. C++ Binary Deployment

For high-performance environments, the C++ binaries can be deployed using `deployment/scripts/deploy_cpp_binary.sh`.

## 4. API Deployment

The REST API or GraphQL service for serving the model in production is provided via:

- **App**: `deployment/api/app.py`
- **Routes**: `deployment/api/routes.py`
