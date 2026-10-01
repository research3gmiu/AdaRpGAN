FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
python3 \
python3-pip \
python3-venv \
git \
wget \
vim \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN pip3 install --upgrade pip

# RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab

# Copy the project files needed for installing dependencies
COPY pyproject.toml README.md ./

# Create a dummy structure to allow pip install -e to work without full source
RUN mkdir -p adarppgan && touch adarppgan/__init__.py

# Install project dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Expose the JupyterLab port
EXPOSE 8888

# Command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
