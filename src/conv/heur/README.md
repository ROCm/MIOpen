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


The following stuff needs to be run inside the docker

### Convert the text find dbs to a pandas data frame. 

These files are typically present in `/opt/rocm/miopen/share/db/miopen/*.fdb.txt`. 

```bash
python parse_finddb.py <path to *.fdb.txt files>  /tmp/find_db.pd
```

If there are multiple files in the target directory all of them would be parsed and collected in the same pandas file. For each architecture in these files, a model will be created.

### Generate onnx models followed by compiled code objects

```bash
python miopen_net.py train --data_filename /tmp/rocm310.pd --save_model assets/
```
where `assets/` already exists

### Generate `metadata.cpp` 

```bash
python miopen_net.py meta --data_filename /tmp/rocm310.pd --output_file metadata.cpp
```

Format the generated file so it passes MIOpen CI

```bash
clang-format-3.8 -i -style=file metadata.cpp
```


