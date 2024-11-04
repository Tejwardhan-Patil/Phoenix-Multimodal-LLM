# API Usage

This document explains how to interact with the Phoenix Multimodal LLM API.

## 1. API Endpoints

The API exposes the following endpoints:

- `/inference`: Accepts multimodal inputs and returns predictions.
- `/health`: Provides the status of the model and server.

## 2. API Routes

The routes are defined in `deployment/api/routes.py`.

## 3. Query

To submit a query for multimodal input (text and image), use the following:

```bash
curl -X POST "https://website.com/inference" \
-H "Content-Type: application/json" \
-d '{
  "text": "Sample text input",
  "image_url": "https://website.com/image.png"
}'
```
