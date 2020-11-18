## Introduction


### Files

model.py Has the heuristics model


## Getting Started

### Build development docker

```bash
docker build -t miopen-heuristics:latest . 
```

The docker builds a local version of the onnx-mlir toolchain as well as installs pytorch in a virtualenv. PyTorch is installed in virtualenv due to conflicts with onnx-mlir dependencies

Launch the docker

```bash
docker run -it -v $HOME:/data miopen-heuristics bash 
```

