FROM onnxmlirczar/onnx-mlir-build:x86@sha256:c8fdeb4f771d13398516e653ecb7ecc1d439c7ac380fc41927f1a4733c2ae092

WORKDIR /build
ENV HOME=/build
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PATH="/build/onnx-mlir/build/bin:/build/llvm-project/build/bin:${PATH}"
RUN pyenv global 3.7.0
RUN pyenv rehash

ENV PATH=$PATH:/build/bin
RUN apt-get update
RUN apt-get install -y python-numpy
RUN apt-get install -y python3-pip
RUN apt-get install -y gdb
RUN apt-get install -y lldb
RUN apt-get install -y vim  clang-format
RUN pip install --upgrade pip
RUN rm -rf /build/onnx-mlir
RUN git clone --recursive -b keep_tmp https://github.com/JehandadKhan/onnx-mlir /build/onnx-mlir
ENV LLVM_PROJ_SRC="/build/llvm-project/"
ENV LLVM_PROJ_BUILD="/build/llvm-project/build"
RUN mkdir /build/onnx-mlir/build && cd /build/onnx-mlir/build && cmake .. && cmake --build .  -- -j $(nproc)

# save off path w/o virtualenv
ENV ONNX_MLIR_VERBOSE=1
ENV OLD_PATH=${PATH}
ENV VIRTUAL_ENV=/build/myvenv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /build
