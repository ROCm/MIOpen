FROM ubuntu:22.04 as miopen

ARG DEBIAN_FRONTEND=noninteractive
# install to /opt/rocm will cause permission issue
ARG PREFIX=/usr/local
ARG USE_FIN="OFF"

ARG CCACHE_SECONDARY_STORAGE=""
ARG CCACHE_DIR="/tmp"
ARG CCACHE_COMMIT=7f1572ae9ca958fa923a66235f6a64a360b03523
ARG MIOPEN_SCCACHE=""
ARG MIOPEN_SCCACHE_CUSTOM_CACHE_BUSTER="MiOpen-Docker-CK"

# GPU_ARCHS should be defined as a build arg rather than hardcoded here.
ARG GPU_ARCHS=none

ARG COMPILER_LAUNCHER=""
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

# Support multiarch
RUN dpkg --add-architecture i386

# Install preliminary dependencies and add rocm gpg key
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
        apt-utils ca-certificates curl libnuma-dev gnupg2 wget  && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg

# Get and install amdgpu-install.
RUN wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/jammy/amdgpu-install_6.4.60403-1_all.deb --no-check-certificate && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
       ./amdgpu-install_6.4.60403-1_all.deb

# Add rocm repository
RUN export ROCM_APT_VER=6.4.3; \
    echo $ROCM_APT_VER &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/amdgpu/$ROCM_APT_VER/ubuntu jammy main > /etc/apt/sources.list.d/amdgpu.list' &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/$ROCM_APT_VER jammy main > /etc/apt/sources.list.d/rocm.list'

RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu jammy main universe | tee -a /etc/apt/sources.list" && \
    amdgpu-install -y --usecase=rocm --no-dkms

## Sccache binary built from source for ROCm, only install if CK_SCCACHE is defined
ARG SCCACHE_REPO_URL=http://compute-artifactory.amd.com/artifactory/rocm-generic-experimental/rocm-sccache
ENV SCCACHE_INSTALL_LOCATION=/usr/local/.cargo/bin
ENV PATH=$PATH:${SCCACHE_INSTALL_LOCATION}
ENV MIOPEN_SCCACHE=$MIOPEN_SCCACHE
RUN if [ "$MIOPEN_SCCACHE" != "" ]; then \
    mkdir -p ${SCCACHE_INSTALL_LOCATION} && \
    curl ${SCCACHE_REPO_URL}/portable/0.2.16/sccache-0.2.16-alpha.1-rocm --output ${SCCACHE_INSTALL_LOCATION}/sccache && \
    chmod +x ${SCCACHE_INSTALL_LOCATION}/sccache; \
    fi

# Add DVC repo
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO - https://dvc.org/deb/iterative.asc | sudo gpg --dearmor -o /etc/apt/keyrings/packages.iterative.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.iterative.gpg] https://dvc.org/deb/ stable main" | sudo tee /etc/apt/sources.list.d/dvc.list && \
    chmod 644 /etc/apt/keyrings/packages.iterative.gpg /etc/apt/sources.list.d/dvc.list

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake \
    clang-format-12 \
    doxygen \
    gdb \
    git \
    git-lfs \
    half \
    lbzip2 \
    lcov \
    libncurses5-dev \
    stunnel \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    redis \
    rocblas-dev \
    rocm-developer-tools \
    rocm-llvm-dev \
    rocrand-dev \
    rpm \
    software-properties-common \
    dvc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf amdgpu-install* && \
# Remove unnecessary rocm components that take a lot of space
    apt-get remove -y miopen-hip

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Add requirements files
ADD rbuild.ini /rbuild.ini
ADD requirements.txt /requirements.txt
ADD dev-requirements.txt /dev-requirements.txt
ADD docs/sphinx/requirements.txt /doc-requirements.txt

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && rm dumb-init_*.deb && \
# Install cget && rbuild
    pip3 install https://github.com/pfultz2/cget/archive/a426e4e5147d87ea421a3101e6a3beca541c8df8.tar.gz && \
    pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/6d78a0553babdaea8d2da5de15cbda7e869594b8.tar.gz && \
# Add symlink to /opt/rocm
    [ -d /opt/rocm ] || ln -sd $(realpath /opt/rocm-*) /opt/rocm && \
# clear ccachc temp
    rm -rf /tmp/ccache* && mkdir /tmp/ccache && \
# Install selected Ccache commit
    wget -O /tmp/ccache.tar.gz https://github.com/ccache/ccache/archive/${CCACHE_COMMIT}.tar.gz && \
    tar zxvf /tmp/ccache.tar.gz -C /tmp/ && mkdir /tmp/ccache-${CCACHE_COMMIT}/build && \
    cd /tmp/ccache-${CCACHE_COMMIT}/build && \
    cmake -DZSTD_FROM_INTERNET=ON -DHIREDIS_FROM_INTERNET=ON .. && make -j install && rm -rf /tmp/* && \
    ccache -s && \
    cd / && \
# Install doc requirements
    pip3 install -r /doc-requirements.txt && \
# Composable Kernel requires this version cmake
    pip3 install --upgrade cmake==3.27.5 && \
# groupadd render && video
    groupadd -f render && \
    groupadd -f video && \
    usermod -a -G render,video root

# Make sure /opt/rocm is in the paths
ENV PATH="/opt/rocm:${PATH}"

ADD script/redis-cli.conf /redis-cli.conf
ADD script/sccache_wrapper.sh /sccache_wrapper.sh

RUN echo Building for GPU Archs: ${GPU_ARCHS} && \
    if [ "$MIOPEN_SCCACHE" != "" ]; then \
    mkdir -p ${SCCACHE_INSTALL_LOCATION} && \
    curl ${SCCACHE_REPO_URL}/portable/0.2.16/sccache-0.2.16-alpha.1-rocm --output ${SCCACHE_INSTALL_LOCATION}/sccache && \
    chmod +x ${SCCACHE_INSTALL_LOCATION}/sccache; \
    export ROCM_PATH=/opt/rocm && \
    export SCCACHE_ENABLED=true && \
    export SCCACHE_LOG_LEVEL=debug && \
    export SCCACHE_IDLE_TIMEOUT=14400 && \
    export COMPILERS_HASH_DIR=/tmp/.sccache && \
    export SCCACHE_BIN=/usr/local/.cargo/bin/sccache && \
    export SCCACHE_EXTRAFILES=/tmp/.sccache/rocm_compilers_hash_file && \
    export SCCACHE_REDIS="redis://$MIOPEN_SCCACHE" && \
    echo "connect = $MIOPEN_SCCACHE" >> redis-cli.conf && \
    export SCCACHE_C_CUSTOM_CACHE_BUSTER="${MIOPEN_SCCACHE_CUSTOM_CACHE_BUSTER}" && \
    echo $SCCACHE_C_CUSTOM_CACHE_BUSTER && \
    stunnel redis-cli.conf && \
    export PATH=$PATH:${SCCACHE_INSTALL_LOCATION} && \
    ./sccache_wrapper.sh --enforce_redis; \
    fi &&\
    CK_COMMIT=$(grep 'ROCm/composable_kernel' requirements.txt | sed -n 's/.*@\([a-zA-Z0-9]*\).*/\1/p') && \
    wget -O ck.tar.gz https://www.github.com/ROCm/composable_kernel/archive/${CK_COMMIT}.tar.gz && \
    tar zxvf ck.tar.gz &&\
    cd composable_kernel-${CK_COMMIT} && \
    mkdir build && cd build && \
    num_threads=64 && \
    echo Building CK with ${num_threads} threads && \
    CXX=/opt/rocm/bin/amdclang++ cmake \
    -D CMAKE_PREFIX_PATH=/opt/rocm \
    -D CMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -D CMAKE_C_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D GPU_ARCHS=${GPU_ARCHS} \
    -D MIOPEN_REQ_LIBS_ONLY=ON \
    -D DISABLE_OFFLOAD_COMPRESS=ON \
    -D CMAKE_CXX_FLAGS=" -O3 " .. && \
    make -j ${num_threads} install && \    
    if [ "$MIOPEN_SCCACHE" != "" ]; then \
    sccache -s; \
    fi

# Composable Kernel installed separated from rbuild to take in values from GPU_ARCHS
RUN sed -i '/composable_kernel/d' /requirements.txt

# rbuild is used to trigger build of requirements.txt, dev-requirements.txt
RUN if [ "$USE_FIN" = "ON" ]; then \
    rbuild prepare -s fin -d $PREFIX -DGPU_ARCHS="${GPU_ARCHS}"; \
    else \
    rbuild prepare -s develop -d $PREFIX -DGPU_ARCHS="${GPU_ARCHS}"; \
    fi && \
    ccache -s

# Utilize multi-stage build in order to squash the container.
FROM ubuntu:22.04
COPY --from=miopen / /
