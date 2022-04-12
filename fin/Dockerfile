FROM ubuntu:18.04

ARG PREFIX=/usr/local

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-utils wget gnupg2
RUN curl https://raw.githubusercontent.com/RadeonOpenCompute/ROCm-docker/master/add-rocm.sh | bash

# Install dependencies required to build hcc
# Ubuntu csomic contains llvm-7 required to build Tensile
RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu xenial main universe | tee -a /etc/apt/sources.list"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    clang-3.9 \
    clang-format-3.9 \
    clang-tidy-3.9 \
    cmake \
    comgr \
    curl \
    doxygen \
    g++-5-multilib \
    git \
    hsa-rocr-dev \
    hsakmt-roct-dev \
    jq \
    lcov \
    libelf-dev \
    libfile-which-perl \
    libncurses5-dev \
    libpthread-stubs0-dev \
    libnuma-dev \
    libunwind-dev \
    nsis \
    python \
    python-dev \
    python-pip \
    software-properties-common \
    libboost-all-dev \
    llvm-7 \
    pkg-config \
    python3 \
    python3-distutils \
    python3-venv \
    python-yaml \
    cppcheck \
    rocm-dev \
    rocm-opencl \
    rocm-opencl-dev \
    rocblas \
    miopen-hip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

# Install cget
RUN pip install https://github.com/pfultz2/cget/archive/57b3289000fcdb3b7e424c60a35ea09bc44d8538.tar.gz

# Install rclone
RUN pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

# Install hcc from ROCm 3.0
RUN rclone -b roc-3.0.x -c 286651a04d9c3a8e3052dd84b1822985498cd27d https://github.com/RadeonOpenCompute/hcc.git /hcc
RUN LDFLAGS=-fuse-ld=gold cget -p $PREFIX install hcc,/hcc  && rm -rf /hcc

# Make sure /opt/rcom is in the paths
ENV PATH="/opt/rocm:${PATH}"

# Build using hcc
RUN cget -p $PREFIX init --cxx $PREFIX/bin/hcc --std=c++14

# Install dependencies
ADD requirements.txt /requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /requirements.txt

# Install doc requirements
#ADD doc/requirements.txt /doc-requirements.txt
#RUN pip install -r /doc-requirements.txt

