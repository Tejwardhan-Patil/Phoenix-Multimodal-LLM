# Model Architectures

This document provides an overview of the various model architectures used within the Phoenix Multimodal LLM system.

## 1. Text Models

- **GPT-like Model**: A text generation and understanding model. Implemented in `models/text/gpt_model.py`.
- **BERT-based Model**: Used for text classification and embeddings. Implemented in `models/text/bert_model.py`.

## 2. Image Models

- **Convolutional Neural Network (CNN)**: Used for image classification tasks. Implemented in `models/image/cnn_model.py`.
- **Vision Transformer (ViT)**: For advanced image processing. Implemented in `models/image/vit_model.py`.

## 3. Audio Models

- **Recurrent Neural Network (RNN)**: Processes audio sequences. Implemented in `models/audio/rnn_audio_model.py`.
- **WaveNet**: A generative model for audio. Implemented in `models/audio/wavenet_model.py`.

## 4. Multimodal Models

- **Text-Image Model**: Combines text and image inputs, similar to CLIP. Implemented in `models/multimodal/text_image_model.py`.
- **Text-Audio Model**: Combines text and audio inputs. Implemented in `models/multimodal/text_audio_model.py`.
- **Fusion Model**: Fuses text, image, and audio features. Implemented in `models/multimodal/fusion_model.py`.

## 5. Pretrained Models

The project includes support for pre-trained models for transfer learning. These models are stored in the `models/pretrained/` directory.
