FROM ubuntu:16.04

ARG PREFIX=/opt/rocm
ARG GITLAB1=10.236.13.205
ARG ARTIFACTORY=172.27.226.104

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-utils wget
RUN curl -sL http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

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

# Install opencl
RUN wget http://$ARTIFACTORY/artifactory/list/deb-experimental-local/amd/rocm/rocm-opencl-dev-1.2.0-1426879_amd64.deb
RUN wget http://$ARTIFACTORY/artifactory/list/deb-experimental-local/amd/rocm/rocm-opencl-1.2.0-1426879_amd64.deb
RUN dpkg -i --force-all rocm-opencl-*.deb && rm rocm-opencl-*.deb

# Install cget
RUN pip install cget

# Add the windows toolchain
ADD cmake/mingw-toolchain.cmake $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake
RUN cget -p $PREFIX/x86_64-w64-mingw32 init -t $PREFIX/x86_64-w64-mingw32/cmake/toolchain.cmake

# Build hcc
RUN git clone --depth 1 https://github.com/RadeonOpenCompute/hcc.git -b hcc-roc-1.5.x /hcc && \
    git clone --depth 1 https://github.com/RadeonOpenCompute/hcc-clang-upgrade.git -b hcc-roc-1.5.x /hcc/clang && \
    git clone --depth 1 https://github.com/RadeonOpenCompute/clang-tools-extra.git -b hcc-roc-1.5.x /hcc/clang/tools/extra && \
    git clone --depth 1 https://github.com/RadeonOpenCompute/llvm.git -b amd-hcc-roc-1.5.x /hcc/compiler && \
    git clone --depth 1 https://github.com/RadeonOpenCompute/compiler-rt.git -b hcc-roc-1.5.x /hcc/compiler-rt && \
    git clone --depth 1 https://github.com/RadeonOpenCompute/lld.git -b hcc-roc-1.5.x /hcc/lld && \
    git clone --depth 1 https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git -b hcc-roc-1.5.x /hcc/rocdl && \
    cget -p $PREFIX install /hcc && rm -rf /hcc

# This is a workaround for broken installations
RUN ln -s $PREFIX /opt/rocm/hip
RUN ln -s $PREFIX /opt/rocm/hcc

# Build using hcc
RUN cget -p $PREFIX init --cxx $PREFIX/bin/hcc

# Install hip
RUN cget -p $PREFIX install ROCm-Developer-Tools/HIP@roc-1.5.0

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
RUN cget -p $PREFIX install -f /dev-requirements.txt

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip install -r /doc-requirements.txt

# Install windows dependencies
RUN cget -p $PREFIX/x86_64-w64-mingw32 install RadeonOpenCompute/rocm-cmake@cb666a28b261fe63ffbcfcf3fee946b1941df604
RUN cget -p $PREFIX/x86_64-w64-mingw32 install -X header meganz/mingw-std-threads@dad05201ad4e096c5d1b2043081f412aeb8f5efb

# Install windows opencl
RUN curl http://$GITLAB1/pfultz/mlopen/uploads/bbab72ad68e65faeee9257b2bb9ca4a1/win-opencl.deb > /win-opencl.deb
RUN dpkg -i /win-opencl.deb && rm /win-opencl.deb

# Setup wine
RUN mkdir -p /jenkins
RUN chmod 777 /jenkins
RUN WINEDEBUG=-all DISPLAY=:55.0 wineboot; wineserver -w
