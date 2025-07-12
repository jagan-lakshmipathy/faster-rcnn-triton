
# Stage 1: Extract Triton server files
FROM nvcr.io/nvidia/tritonserver:22.12-py3 AS triton

# Stage 2: Your base CUDA + Ubuntu + PyTorch etc.
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.10-venv \
        curl \
        git \
        cmake \
        sudo \
        wget \
        build-essential \
        libgl1 \
        libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic Python build tools
RUN python -m pip install --upgrade pip setuptools wheel ninja

COPY cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz /tmp/
RUN tar -xvf /tmp/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz -C /tmp && \
    cp -P /tmp/cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include/* /usr/local/cuda/include/ && \
    cp -P /tmp/cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib/* /usr/local/cuda/lib64/ && \
    ldconfig && \
    rm -rf /tmp/cudnn*
ENV CUDNN_DIR=/home/atharva/cudnn-linux-x86_64-8.5.0.96_cuda11-archive
ENV LD_LIBRARY_PATH=$CUDNN_DIR/lib:$LD_LIBRARY_PATH

# Install PyTorch 2.0.0 + CUDA 11.8 and torchvision 0.15.1 + torchaudio 2.0.1
RUN pip install --no-cache-dir torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN pip install -U openmim

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Clone and build MMCV v2.0.0
RUN git clone -b v2.0.0 https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    pip install -r requirements/build.txt && \
    pip install -v -e .

RUN pip install mmengine==0.7.1
RUN pip install mmdet==3.0.0

# Install dependencies for MMDeploy
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotoc-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libopencv-dev

# Clone and build MMDeploy v1.2.0
RUN git clone -b v1.2.0 https://github.com/open-mmlab/mmdeploy.git && \
    cd mmdeploy && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    mkdir -p build && cd build && \
    cmake .. -DMMDEPLOY_BUILD_SDK=ON \
             -DMMDEPLOY_BUILD_EXAMPLES=OFF \
             -DMMDEPLOY_TARGET_BACKENDS="onnxruntime" \
             -DMMDEPLOY_CODEBASES="mmdet" && \
    make -j$(nproc) && \
    cd .. && \
    pip install -v -e .


RUN pip install git+https://github.com/PaddlePaddle/PaddleOCR.git
RUN pip install tqdm loguru paddlepaddle-gpu

RUN pip install --no-cache-dir openxlab==0.1.2 requests==2.28.2 pandas==1.5.3 \
    numpy==1.24.4 matplotlib==3.7.2 tqdm==4.65.0 packaging>=22.0 pytz==2023.3 rich==13.4.2 \
    networkx==3.1 jupyterlab notebook ipykernel

# Download and install Triton client libraries (v2.34.0) from official release
RUN curl -L -o /tmp/triton_clients.tar.gz https://github.com/triton-inference-server/server/releases/download/v2.34.0/v2.34.0_ubuntu2004.clients.tar.gz && \
    mkdir /tmp/v2.34.0 && \
    tar -xzf /tmp/triton_clients.tar.gz -C /tmp/v2.34.0 && \
    pip install /tmp/v2.34.0/python/tritonclient-2.34.0-py3-none-manylinux1_x86_64.whl && \
    pip install "tritonclient[http]" && \
    pip install gevent requests urllib3 && \
    rm -rf /tmp/triton_clients.tar.gz /tmp/v2.34.0

# ✅ Copy Triton server files from Stage 1
COPY --from=triton /opt/tritonserver /opt/tritonserver

# Copy libdcgm.so.2 from Triton image
COPY --from=triton /usr/lib/x86_64-linux-gnu/libdcgm.so.2 /usr/lib/x86_64-linux-gnu/

# Add Triton to PATH and LD_LIBRARY_PATH
ENV PATH="/opt/tritonserver/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/tritonserver/lib:${LD_LIBRARY_PATH}"


RUN pip install triton-model-analyzer==1.20.0

RUN apt-get update && apt-get install -y libre2-dev && \
    apt-get update && apt-get install -y libb64-dev

# Set working directory
WORKDIR /workspace

# Expose port for JupyterLab
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
