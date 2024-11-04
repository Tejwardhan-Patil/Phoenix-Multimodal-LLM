# Multimodal Learning

This document provides a detailed overview of the multimodal learning techniques and strategies used in the Phoenix Multimodal LLM.

## 1. Modalities Supported

- **Text**: Preprocessed through tokenization and embedding methods.
- **Image**: Processed through CNN or Vision Transformer models.
- **Audio**: Handled by RNN and WaveNet models.

## 2. Fusion Techniques

The fusion of multiple modalities is handled through the fusion model implemented in `models/multimodal/fusion_model.py`. This model combines features from text, images, and audio inputs.

## 3. Training Strategy

The training is performed using `models/train.py`, with different loss functions applied to each modality. Cross-modality evaluations are implemented in `models/evaluate.py`.

## 4. Inference Pipeline

Inference is done via the script `models/inference.py`, which can accept multimodal inputs and provide predictions. Optimized C++ inference is available via `models/inference_cpp.cpp`.

## 5. Multimodal Alignment and Evaluation

The evaluation of multimodal coherence and alignment metrics is integrated within `experiments/results/` and tracked via `monitoring/metrics/`.
