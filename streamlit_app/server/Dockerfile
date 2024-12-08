# FastAPI Dockerfile
FROM python:3.10-slim

# Set environment variables to ensure Python runs optimally
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /server

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
        llvm \
        llvm-dev \
        clang \
        build-essential \
        curl \
        wget \
        software-properties-common \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set LLVM_CONFIG to the compatible version
ENV LLVM_CONFIG=/usr/lib/llvm-10/bin/llvm-config

# Pre-install numpy to avoid numba's numpy import issue
RUN pip install --upgrade pip
RUN pip install numpy==2.1.3

# Install Python dependencies from requirements
COPY server_requirements.txt /server/
RUN pip install --no-cache-dir -r server_requirements.txt

# Copy application files
COPY . /server/

# Expose the necessary port (default for FastAPI)
EXPOSE 8081

# Command to run the FastAPI server
CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8081"]