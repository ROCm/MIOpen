FROM ubuntu:16.04

ARG PREFIX=/opt/rocm
ARG GITLAB1=10.236.13.205

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl && \
    curl -sL http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

# Install dependencies required to build hcc
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    clang-3.8 \
    clang-format-3.8 \
    clang-tidy-3.8 \
    cmake \
    curl \
    g++-mingw-w64 \
    g++-mingw-w64-x86-64 \
    g++-multilib \
    git \
    hsa-rocr-dev \
    hsakmt-roct-dev \
    libelf-dev \
    libncurses5-dev \
    libpthread-stubs0-dev \
    mingw-w64 \
    mingw-w64-tools \
    nsis \
    python \
    python-dev \
    python-pip \
    software-properties-common \
    wget \
    wine \
    xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

# Install cget
RUN pip install cget

# Install latest cmake
RUN cget -p /usr/local install kitware/cmake@release

# Add the toolchain
ADD cmake/mingw-toolchain.cmake /usr/local/x86_64-w64-mingw32/cmake/toolchain.cmake

# Build hcc
RUN git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git /hcc && \
    git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc-clang-upgrade.git /hcc/clang && \
    git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/clang-tools-extra.git /hcc/clang/tools/extra && \
    git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/llvm.git /hcc/compiler && \
    git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/compiler-rt.git /hcc/compiler-rt && \
    git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/lld.git /hcc/lld && \
    git clone --depth 1 -b remove-promote-change-addr-space https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git /hcc/rocdl && \
    cget -p $PREFIX install hcc,/hcc && \
    rm -rf /hcc

# Not needed by miopen, but it helps downstream applications
RUN ln -s $PREFIX $PREFIX/hip
RUN ln -s $PREFIX $PREFIX/hcc

# Build using hcc
RUN cget -p $PREFIX init --cxx $PREFIX/bin/hcc

# Install HIP
RUN cget -p $PREFIX install hip,http://$GITLAB1/pfultz/hip/repository/archive.tar.gz?ref=cmake-develop

# Install opencl
RUN curl http://$GITLAB1/pfultz/mlopen/uploads/194a8f592aaeabb486e3594e3a4083e6/rocm-opencl-1.4.deb > /rocm-opencl.deb
RUN dpkg -i /rocm-opencl.deb && rm /rocm-opencl.deb

# Install clang-ocl
RUN cget -p $PREFIX install clang-ocl,http://$GITLAB1/pfultz/clang-ocl/repository/archive.tar.bz2?ref=master

# Install tinygemm
RUN cget -p /usr/local install tinygemm,http://$GITLAB1/pfultz/tinygemm/repository/archive.tar.gz?ref=master

# Install windows opencl
RUN curl http://$GITLAB1/pfultz/mlopen/uploads/bbab72ad68e65faeee9257b2bb9ca4a1/win-opencl.deb > /win-opencl.deb
RUN dpkg -i /win-opencl.deb && rm /win-opencl.deb

# Install mingw threads
RUN cget -p /usr/local/x86_64-w64-mingw32 install -X header meganz/mingw-std-threads@master

# Setup wine
RUN WINEDEBUG=-all DISPLAY=:55.0 wineboot; wineserver -w
