FROM ubuntu:18.04

ARG PREFIX=/opt/rocm

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-utils wget gnupg2

#RUN curl https://raw.githubusercontent.com/RadeonOpenCompute/ROCm-docker/master/add-rocm.sh | bash
ARG ROCMVERSION=4.5
ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN sh -c "echo deb [arch=amd64] $DEB_ROCM_REPO ubuntu main > /etc/apt/sources.list.d/rocm.list"

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

# Make sure /opt/rcom is in the paths
ENV PATH="/opt/rocm:${PATH}"
# Install MIOpen
ARG MIOPEN_DIR=/root/dMIOpen
#Clone MIOpen
RUN git clone https://github.com/ROCmSoftwarePlatform/MIOpen.git $MIOPEN_DIR
WORKDIR $MIOPEN_DIR

ARG MIOPEN_DEPS=$MIOPEN_DIR/cget
# Install dependencies
RUN wget --no-check-certificate -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN sh -c "echo deb https://apt.kitware.com/ubuntu/ bionic main | tee -a /etc/apt/sources.list"
RUN apt-get update && apt-get install -y --allow-unauthenticated   cmake-data=3.15.1-0kitware1 cmake=3.15.1-0kitware1
RUN cget install pfultz2/rocm-recipes
RUN cget install -f min-requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget install -f ./mlir-requirements.txt

ARG TUNA_USER=miopenpdb
ARG BACKEND=HIP
# Build MIOpen
WORKDIR $MIOPEN_DIR/build
ARG MIOPEN_BRANCH=05169b091cf3d0b752c94ba51d63f7af6463d731
ARG MIOPEN_CACHE_DIR=/tmp/${TUNA_USER}/cache
ARG MIOPEN_USER_DB_PATH=/tmp/$TUNA_USER/config/miopen
ARG MIOPEN_USE_MLIR=On
RUN git pull && git checkout $MIOPEN_BRANCH
RUN echo "MIOPEN: Selected $BACKEND backend."
RUN if [ $BACKEND = "OpenCL" ]; then \
           cmake -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_USE_COMGR=Off -DMIOPEN_USE_MLIR=${MIOPEN_USE_MLIR} -DMIOPEN_CACHE_DIR=${MIOPEN_CACHE_DIR} -DMIOPEN_USER_DB_PATH={MIOPEN_USER_DB_PATH} -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$MIOPEN_DEPS" $MIOPEN_DIR ; \
    else \
           CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_USE_COMGR=Off -DMIOPEN_USE_MLIR=${MIOPEN_USE_MLIR} -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_CACHE_DIR=${MIOPEN_CACHE_DIR} -DMIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH} -DMIOPEN_BACKEND=$BACKEND -DCMAKE_PREFIX_PATH=$MIOPEN_DEPS $MIOPEN_DIR ; \
    fi

RUN make -j $(nproc)
RUN make install

# Install dependencies
ADD requirements.txt /requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /requirements.txt
