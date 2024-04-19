.. meta::
  :description: Build MIOpen for embedded systems
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
Build MIOpen for embedded systems
********************************************************************

1. Install dependencies. The default location is ``/usr/local``:

  .. code:: cpp

    cmake -P install_deps.cmake --minimum --prefix /some/local/dir

2. Create the build directory.

  .. code:: cpp

    mkdir build; cd build;

3. Configure for an embedded build.

  The minimum static build configuration line, without an embedded precompiled kernels package or
  FindDb is:

  .. code:: cpp

    CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_EMBED_BUILD=On -DCMAKE_PREFIX_PATH="/some/local/dir" ..


  To enable HIP kernels in MIOpen while using embedded builds, add
  ``-DMIOPEN_USE_HIP_KERNELS=On`` to configure line. For example:

  .. code:: cpp

    CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_USE_HIP_KERNELS=On -DMIOPEN_EMBED_BUILD=On -DCMAKE_PREFIX_PATH="/some/local/dir" ..

4. Embed FindDb and PerfDb.

  FindDb provides a database of known convolution inputs. This allows you to use the best tuned
  kernels for your network. Embedding FindDb requires a semicolon-separated list of architecture CU
  pairs to embed on-disk databases in the binary (e.g., ``gfx906_60;gfx900_56``).

  .. code:: cpp

    CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_EMBED_BUILD=On -DMIOPEN_EMBED_DB=gfx900_56 ..

  This configures the build directory for embedding (FindDb and PerfDb).

5. Embed the precompiled kernels package.

  To prevent the loss of performance due to compile-time overhead, an MIOpen build can embed the
  precompiled kernels package. This package contains convolution kernels of known inputs, and allows
  you to avoid compiling kernels during runtime.

  * Embed the precompiled package using a package install.

    .. code:: bash

      apt-get install miopenkernels-<arch>-<num cu>

    Where ``<arch>`` is the GPU architecture (e.g., gfx900, gfx906) and ``<num cu>`` is the number of
    compute units (CUs) available in the GPU (e.g., 56, 64).

    If you choose not to install the precompiled kernel package, there is no impact to the functioning of
    MIOpen because MIOpen compiles these kernels on the target machine once the kernel is run.
    However, the compilation step may significantly increase the startup time for different operations.

    The ``utils/install_precompiled_kernels.sh`` script automates the above process. It queries your
    machine for the GPU architecture and then installs the appropriate package. You can invoke it using:

    .. code:: cpp

      ./utils/install_precompiled_kernels.sh

    To embed the precompiled kernels package, configure CMake using the
    ``MIOPEN_BINCACHE_PATH``.

    .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/path/to/package/install -DMIOPEN_EMBED_BUILD=On ..

  * Using the URL to a kernels binary. You can use the ``MIOPEN_BINCACHE_PATH`` flag with a URL that
    contains the binary.

    .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/URL/to/binary -DMIOPEN_EMBED_BUILD=On ..

    Precompiled kernels packages are installed in ``/opt/rocm/miopen/share/miopen/db``.

    Here's an example with gfx900 architecture and 56 CUs:

    .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/opt/rocm/miopen/share/miopen/db/gfx900_56.kdb -DMIOPEN_EMBED_BUILD=On ..

    As of ROCm 3.8 and MIOpen 2.7, precompiled kernels binaries are located at
    `repo.radeon.com <http://repo.radeon.com/rocm/miopen-kernel/>`_.

    Here's an example with gfx906 architecture and 64 CUs:

    .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=http://repo.radeon.com/rocm/miopen-kernel/rel-3.8/gfx906_60.kdb -DMIOPEN_EMBED_BUILD=On .. 


6. Full configuration line.

  To build MIOpen statically and embed the performance database, FindDb, and the precompiled
  kernels binary:

  .. code:: cpp

    CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/path/to/package/install -DMIOPEN_EMBED_BUILD=On -DMIOPEN_EMBED_DB=gfx900_56 ..

  After configuration is complete, run:

  .. code:: cpp

    make -j
