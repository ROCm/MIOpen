.. meta::
  :description: Build MIOpen for embedded systems
  :keywords: MIOpen, ROCm, API, documentation, embedded, build

********************************************************************
Build MIOpen for embedded systems
********************************************************************

Follow these steps to build and configure MIOpen for an embedded system.

.. note::

   You can run ``install_deps.cmake`` from the ``rocm-libraries/projects/miopen`` directory.

1. Install the dependencies. The default install location is ``/usr/local``:

   .. code:: cpp

      cmake -P install_deps.cmake --minimum --prefix /some/local/dir

2. Create the build directory.

   .. code:: cpp

      mkdir build; cd build;

3. Add the embedded build configuration.

   The minimum static build configuration, without an
   embedded precompiled kernels package or
   FindDb, is the following:

   .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_EMBED_BUILD=On -DCMAKE_PREFIX_PATH="/some/local/dir" ..

   To enable HIP kernels in MIOpen for embedded builds, add
   ``-DMIOPEN_USE_HIP_KERNELS=On`` to the command line. For example:

   .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_USE_HIP_KERNELS=On -DMIOPEN_EMBED_BUILD=On -DCMAKE_PREFIX_PATH="/some/local/dir" ..

4. Embed FindDb and PerfDb.

   FindDb provides a database of known convolution inputs. It allows you to use the best tuned
   kernels for your network. To embed on-disk databases in the binary with FindDb, use ``DMIOPEN_EMBED_DB`` with
   a semicolon-separated list of architecture-CU pairs (for example, ``gfx906_60;gfx900_56``).

   .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_EMBED_BUILD=On -DMIOPEN_EMBED_DB=gfx900_56 ..

   This configures embedding for the build directory for both FindDb and PerfDb.

5. Embed the precompiled kernels package.

   An MIOpen build can embed the precompiled kernels package,
   preventing reduced performance due to compile-time overhead. This package contains the convolution kernels of known inputs
   to avoid the runtime compilation of kernels.

   There are two options for embedding the precompiled kernels package.

   *  Embed the precompiled package using a package install.

      .. code:: bash

         apt-get install miopenkernels-<arch>-<num cu>

      Where ``<arch>`` is the GPU architecture (for example, ``gfx900`` or ``gfx906``) and ``<num cu>`` is the number of
      compute units (CUs) available in the GPU (for example, 56 or 64).

      There is no functional impact to MIOpen if you choose not to install the
      precompiled kernel package. This is because MIOpen compiles these kernels on the target machine after the kernel is run.
      However, the compilation step might significantly increase the startup time for some operations.

      The ``utils/install_precompiled_kernels.sh`` script automates this process. It queries your
      system for the GPU architecture and then installs the appropriate package. To invoke it, use:

      .. code:: cpp

         ./utils/install_precompiled_kernels.sh

      To embed the precompiled kernels package, configure CMake using ``MIOPEN_BINCACHE_PATH``.

      .. code:: cpp

         CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/path/to/package/install -DMIOPEN_EMBED_BUILD=On ..

      Here's an example that uses the gfx900 architecture and 56 CUs:

      .. code:: cpp

         CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/opt/rocm/miopen/share/miopen/db/gfx900_56.kdb -DMIOPEN_EMBED_BUILD=On ..

   *  Embed the precompiled package using the URL of a kernels binary. Use the ``MIOPEN_BINCACHE_PATH`` flag with the URL
      of the binary.

      .. code:: cpp

         CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/URL/to/binary -DMIOPEN_EMBED_BUILD=On ..

      The precompiled kernels packages are installed in ``/opt/rocm/miopen/share/miopen/db``.

      As of ROCm version 3.8 and MIOpen version 2.7, precompiled kernels binaries are located at
      `repo.radeon.com <http://repo.radeon.com/rocm/miopen-kernel/>`_.

      Here's an example that uses the gfx906 architecture and 64 CUs:

      .. code:: cpp

         CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=http://repo.radeon.com/rocm/miopen-kernel/rel-3.8/gfx906_60.kdb -DMIOPEN_EMBED_BUILD=On ..

6. Full configuration line.

   To build MIOpen statically and embed the performance database, FindDb, and the precompiled
   kernels binary, follow this example:

   .. code:: cpp

      CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BINCACHE_PATH=/path/to/package/install -DMIOPEN_EMBED_BUILD=On -DMIOPEN_EMBED_DB=gfx900_56 ..

   After configuration is complete, run the following command:

   .. code:: cpp

      make -j
