FROM ubuntu:16.04

ARG PREFIX=/opt/rocm

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-utils wget
RUN curl https://raw.githubusercontent.com/RadeonOpenCompute/ROCm-docker/develop/add-rocm.sh | bash

# Install dependencies required to build hcc
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    clang-3.8 \
    clang-format-3.8 \
    clang-tidy-3.8 \
    cmake \
    curl \
    doxygen \
    g++-mingw-w64 \
    g++-mingw-w64-x86-64 \
    g++-multilib \
    git \
    hsa-rocr-dev \
    hsakmt-roct-dev \
    lcov \
    libelf-dev \
    libncurses5-dev \
    libpthread-stubs0-dev \
    libunwind-dev \
    mingw-w64 \
    mingw-w64-tools \
    nsis \
    python \
    python-dev \
    python-pip \
    rocm-opencl \
    rocm-opencl-dev \
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

# Add the windows toolchain
ADD cmake/mingw-toolchain.cmake $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake
RUN cget -p $PREFIX/x86_64-w64-mingw32 init -t $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake

# Build hcc
RUN git clone --depth 1 https://github.com/RadeonOpenCompute/hcc.git -b roc-1.7.x /hcc && \
    cd hcc && \
    git submodule init && \
    git submodule update --recursive && \
    cget -p $PREFIX install hcc,. && cd .. && rm -rf /hcc

# This is a workaround for broken installations
RUN ln -s $PREFIX /opt/rocm/hip
RUN ln -s $PREFIX /opt/rocm/hcc

# Build using hcc
RUN cget -p $PREFIX init --cxx $PREFIX/bin/hcc

# Install cppcheck
RUN cget -p $PREFIX install danmar/cppcheck@1.81

# Install hip
RUN cget -p $PREFIX install ROCm-Developer-Tools/HIP@85975e719dcae7dbd4d0edf6e691c197c0a4f18c

RUN cget -p $PREFIX install pfultz2/rocm-recipes

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /dev-requirements.txt
RUN cget -p $PREFIX install RadeonOpenCompute/clang-ocl@2f118b5b6b05f0b17467ef07a8bd3b8e5d8b3aac

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip install -r /doc-requirements.txt

# Install windows opencl
RUN cget -p $PREFIX/x86_64-w64-mingw32/opencl init -t $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake
RUN cget install -p $PREFIX/x86_64-w64-mingw32/opencl KhronosGroup/OpenCL-Headers@master -X header -DINCLUDE_DIR=opencl22
RUN cget install -p $PREFIX/x86_64-w64-mingw32/opencl pfultz2/OpenCL-ICD-Loader@master

# Install windows dependencies
RUN cget -p $PREFIX/x86_64-w64-mingw32 install pfultz2/rocm-recipes
RUN cget -p $PREFIX/x86_64-w64-mingw32 install -X header meganz/mingw-std-threads@dad05201ad4e096c5d1b2043081f412aeb8f5efb
RUN ln -s $PREFIX/x86_64-w64-mingw32/include/mingw.thread.h $PREFIX/x86_64-w64-mingw32/include/thread 
# RUN CXXFLAGS='-I $PREFIX/x86_64-w64-mingw32/include' AMDAPPSDKROOT=$PREFIX/x86_64-w64-mingw32/opencl cget -p $PREFIX/x86_64-w64-mingw32 install -f /dev-requirements.txt

# Setup wine
RUN mkdir -p /jenkins
RUN chmod 777 /jenkins
RUN WINEDEBUG=-all DISPLAY=:55.0 wineboot; wineserver -w
