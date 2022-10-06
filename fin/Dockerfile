FROM ubuntu:20.04

ARG PREFIX=/opt/rocm

# Support multiarch
RUN dpkg --add-architecture i386


#install rocm
ARG ROCMVERSION=5.1
ARG OSDB_BKC_VERSION
ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
# Add rocm repository
RUN apt-get update
RUN apt-get install -y wget gnupg
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN if ! [ -z $OSDB_BKC_VERSION ]; then \
       echo "Using BKC VERISION: $OSDB_BKC_VERSION";\
       sh -c "echo deb [arch=amd64 trusted=yes] http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-20.04-deb/ compute-rocm-dkms-no-npi-hipclang ${OSDB_BKC_VERSION} > /etc/apt/sources.list.d/rocm.list" ;\
       cat  /etc/apt/sources.list.d/rocm.list;\
    else \
       sh -c "echo deb [arch=amd64] $DEB_ROCM_REPO ubuntu main > /etc/apt/sources.list.d/rocm.list" ;\
    fi


# Install dependencies required to build hcc
# Ubuntu csomic contains llvm-7 required to build Tensile
RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu xenial main universe | tee -a /etc/apt/sources.list"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -f -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    clang \
    clang-format \
    clang-tidy \
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
    software-properties-common \
    libboost-all-dev \
    llvm-7 \
    pkg-config \
    python3 \
    python3-distutils \
    python3-venv \
    python3-dev \
    python3-pip \
    python3-yaml \
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
ARG MIOPEN_BRANCH=develop
RUN git pull && git checkout $MIOPEN_BRANCH

# Install dependencies
ARG MIOPEN_DEPS=$MIOPEN_DIR/cget
#issue with upstream for composable kernel install
RUN sed -i "s#[^\n]*composable_kernel[^\n]*##g" requirements.txt
RUN cmake -P install_deps.cmake --minimum

ARG TUNA_USER=miopenpdb
ARG BACKEND=HIP
# Build MIOpen
WORKDIR $MIOPEN_DIR/build
ARG MIOPEN_CACHE_DIR=/tmp/${TUNA_USER}/cache
ARG MIOPEN_USER_DB_PATH=/tmp/$TUNA_USER/config/miopen
ARG MIOPEN_USE_MLIR=On
ARG MIOPEN_CMAKE_ARGS="-DMIOPEN_USE_COMGR=Off -DMIOPEN_USE_MLIR=${MIOPEN_USE_MLIR} -DMIOPEN_INSTALL_CXX_HEADERS=On -DMIOPEN_CACHE_DIR=${MIOPEN_CACHE_DIR} -DMIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH} -DMIOPEN_BACKEND=${BACKEND} -DCMAKE_PREFIX_PATH=${MIOPEN_DEPS} -DMIOPEN_USE_COMPOSABLEKERNEL=Off -DUSE_FIN=Off"

RUN echo "MIOPEN: Selected $BACKEND backend."
RUN if [ $BACKEND = "OpenCL" ]; then \
        cmake -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ ${MIOPEN_CMAKE_ARGS} $MIOPEN_DIR ; \
    else \
        CXX=/opt/rocm/llvm/bin/clang++ cmake ${MIOPEN_CMAKE_ARGS} $MIOPEN_DIR ; \
    fi

RUN make -j $(nproc)
RUN make install

# Install dependencies
ADD requirements.txt /requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /requirements.txt
