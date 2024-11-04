# Data Pipeline

This document outlines the data pipeline for the Phoenix Multimodal LLM, from raw data ingestion to feature extraction.

## 1. Raw Data

- **Text Data**: Located in `data/raw/text/`.
- **Image Data**: Located in `data/raw/images/`.
- **Audio Data**: Located in `data/raw/audio/`.

## 2. Preprocessing Scripts

- **Text Preprocessing**: Tokenization, stopword removal, and normalization using `data/scripts/preprocess_text.py`.
- **Image Preprocessing**: Resizing and normalization using `data/scripts/preprocess_images.py`.
- **Audio Preprocessing**: Spectrogram conversion and noise reduction using `data/scripts/preprocess_audio.py`.

## 3. Data Augmentation

The augmentation scripts for all modalities are located in `data/scripts/augmentation.py`.

## 4. Processed Data

The preprocessed data is stored in `data/processed/`, with features stored in `data/features/`.

## 5. C++ Optimizations

For performance-critical tasks, C++ implementations exist:

- **Audio Preprocessing**: `data/scripts/preprocess_audio_cpp.cpp`.
- **Image Preprocessing**: `data/scripts/image_preprocessing_cpp.cpp`.
