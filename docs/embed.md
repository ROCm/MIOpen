
Building MIOpen for Embedded Systems
====================================



### Install dependencies
Install minimum dependencies (default location /usr/local):
```
cmake -P install_deps.cmake --minimum --prefix /some/local/dir
```

Create build directory:
```
mkdir build; cd build;
```

### Configuring for an embedded build
Minimal static build configuration line without embedded precompiled kernels package, or Find-Db:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_EMBED_BUILD=On -DCMAKE_PREFIX_PATH="/some/local/dir" ..
```

To enable HIP kernels in MIOpen while using embedded builds add: `-DMIOPEN_USE_HIP_KERNELS=On` to the configure line.
For example:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_USE_HIP_KERNELS=On -DMIOPEN_EMBED_BUILD=On -DCMAKE_PREFIX_PATH="/some/local/dir" ..
```


### Embedding Find-Db and Performance database:
The Find-db provides a database of known convolution inputs. This allows user to have the best tuned kernels for their network. Embedding find-db requires a semi-colon separated list of architecture CU pairs to embed on-disk DBs in the binary; e.g., gfx906_60;gfx900_56.

Example:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_EMBED_BUILD=On -DMIOPEN_EMBED_DB=gfx900_56 ..
```

This will configure the build directory for embedding not just the find-db, but also the performance database. 

### Embedding the precompiled kernels package:
To prevent the loss of performance due to compile time overhead, a build of MIOpen can take advantage of embedding the precompiled kernels package. The precompiled kernels package contains convolution kernels of known inputs and allows the user to avoid compiling kernels during runtime.

### Embedding precompiled package

#### Using a package install
To install the precompiled kernels package use the command:
```
apt-get install miopenkernels-<arch>-<num cu>
```
Where `<arch>` is the GPU architecture (for example, gfx900, gfx906) and `<num cu>` is the number of CUs available in the GPU (for example 56 or 64 etc).

Not installing the precompiled kernel package would not impact the functioning of MIOpen, since MIOpen will compile these kernels on the target machine once the kernel is run, however, the compilation step may significantly increase the startup time for different operations.

The script `utils/install_precompiled_kernels.sh` provided as part of MIOpen automates the above process, it queries the user machine for the GPU architecture and then installs the appropriate package. It may be invoked as:
```
./utils/install_precompiled_kernels.sh
```

To embed the precompiled kernels package, configure cmake using the `MIOPEN_BINCACHE_PATH`
Example:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/path/to/package/install -DMIOPEN_EMBED_BUILD=On .. 
```

#### Using the URL to a kernels binary 
Alternatively, the flag `MIOPEN_BINCACHE_PATH` can be used with a URL that contains the binary.
Example:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/URL/to/binary -DMIOPEN_EMBED_BUILD=On .. 
```

Precompiled kernels packages are installed in `/opt/rocm/miopen/share/miopen/db`.
An example with the architecture gfx900 with 56 compute units:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/opt/rocm/miopen/share/miopen/db/gfx900_56.kdb -DMIOPEN_EMBED_BUILD=On .. 
```


As of ROCm 3.8 / MIOpen 2.7 precompiled kernels binaries are located at [repo.radeon.com](http://repo.radeon.com/rocm/miopen-kernel/)
For example for the architecture gfx906 with 64 compute units:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=http://repo.radeon.com/rocm/miopen-kernel/rel-3.8/gfx906_60.kdb -DMIOPEN_EMBED_BUILD=On .. 
```

### Full configuration line:
Putting it all together, building MIOpen statically, and embedding the performance database, find-db, and the precompiled kernels binary:
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/path/to/package/install -DMIOPEN_EMBED_BUILD=On -DMIOPEN_EMBED_DB=gfx900_56 .. 
```

After configuration is complete, run:
```
make -j
```





