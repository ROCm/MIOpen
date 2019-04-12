FROM ubuntu:16.04

ARG PREFIX=/opt/rocm

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-utils wget
RUN curl https://raw.githubusercontent.com/RadeonOpenCompute/ROCm-docker/master/add-rocm.sh | bash

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
    libnuma-dev \
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

# Setup ubsan environment to printstacktrace
RUN ln -s /usr/bin/llvm-symbolizer-3.8 /usr/local/bin/llvm-symbolizer
ENV UBSAN_OPTIONS=print_stacktrace=1

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

# Install cget
RUN pip install https://github.com/pfultz2/cget/archive/57b3289000fcdb3b7e424c60a35ea09bc44d8538.tar.gz

# Add the windows toolchain
ADD cmake/mingw-toolchain.cmake $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake
RUN cget -p $PREFIX/x86_64-w64-mingw32 init -t $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake

# Install rclone
RUN pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

# Install hcc
RUN rclone -b roc-2.1.x  -c fea3e2b4625f55b40b8c51c6e445117b3f16166f https://github.com/RadeonOpenCompute/hcc.git /hcc
RUN cget -p $PREFIX install hcc,/hcc  && rm -rf /hcc

# This is a workaround for broken installations
RUN ln -s $PREFIX /opt/rocm/hip
RUN ln -s $PREFIX /opt/rocm/hcc

# Build using hcc
RUN cget -p $PREFIX init --cxx $PREFIX/bin/hcc --std=c++14

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
ADD min-requirements.txt /min-requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /dev-requirements.txt

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip install -r /doc-requirements.txt

# Install windows opencl
RUN cget -p $PREFIX/x86_64-w64-mingw32/opencl init -t $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake
# RUN cget install -p $PREFIX/x86_64-w64-mingw32/opencl KhronosGroup/OpenCL-Headers@master -X header -DINCLUDE_DIR=opencl22
# RUN cget install -p $PREFIX/x86_64-w64-mingw32/opencl pfultz2/OpenCL-ICD-Loader@master

# Install windows dependencies
RUN cget -p $PREFIX/x86_64-w64-mingw32 install pfultz2/rocm-recipes
RUN cget -p $PREFIX/x86_64-w64-mingw32 install -X header meganz/mingw-std-threads@dad05201ad4e096c5d1b2043081f412aeb8f5efb
RUN ln -s $PREFIX/x86_64-w64-mingw32/include/mingw.thread.h $PREFIX/x86_64-w64-mingw32/include/thread 
# RUN CXXFLAGS='-I $PREFIX/x86_64-w64-mingw32/include' AMDAPPSDKROOT=$PREFIX/x86_64-w64-mingw32/opencl cget -p $PREFIX/x86_64-w64-mingw32 install -f /requirements.txt

# Setup wine
RUN mkdir -p /jenkins
RUN chmod 777 /jenkins
RUN WINEDEBUG=-all DISPLAY=:55.0 wineboot; wineserver -w
