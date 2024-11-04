# Base image with Python and C++ dependencies
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev python3-venv \
    build-essential cmake git curl libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev libopencv-dev libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
RUN python3 -m pip install --upgrade pip \
    && pip install -r deployment/api/requirements.txt

# Compile C++ components
RUN mkdir -p build && cd build \
    && cmake .. && make -j$(nproc)

# Expose API port
EXPOSE 8000

# Run the API service using gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8000", "deployment.api.app:app", "--workers", "4", "--timeout", "120"]