.. meta::
  :description: Installing MIOpen
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
Installing MIOpen
********************************************************************

To install MIOpen, you must first install these prerequisites:

* A :doc:`ROCm <rocm:index>`-enabled platform
* A base software stack that includes either:

  * HIP (HIP and HCC libraries and header files)
  * OpenCL (OpenCL libraries and header files)--this is now deprecated

* `ROCm CMake <https://github.com/ROCm/rocm-cmake>`_: provides CMake modules for common
  build tasks needed for the ROCm software stack
* `Half <http://half.sourceforge.net/>`_: IEEE 754-based, half-precision floating-point library
* `Boost <http://www.boost.org/>`_: Version 1.79 is recommended, as older versions may need patches
  to work on newer systems

  * MIOpen uses ``boost-system`` and ``boost-filesystem`` packages to enable persistent
    :doc:`kernel cache <../conceptual/cache>`

* `SQLite3 <https://sqlite.org/index.html>`_: A reading and writing performance database
* lbzip2: A multi-threaded compress or decompress utility
* :doc:`rocBLAS <rocblas:index>`: AMD's library for Basic Linear Algebra Subprograms (BLAS) on the
  ROCm platform.

  * Minimum version branch for pre-ROCm 3.5
    `master-rocm-2.10 <https://github.com/ROCm/rocBLAS/tree/master-rocm-2.10>`_
  * Minimum version branch for post-ROCm 3.5
    `master-rocm-3.5 <https://github.com/ROCm/rocBLAS/tree/master-rocm-3.5>`_

* `Multi-Level Intermediate Representation (MLIR) <https://github.com/ROCm/rocMLIR>`_ with its
  MIOpen dialect to support and complement kernel development
* :doc:`Composable Kernel <composable_kernel:index>`: A C++ templated device library for
  GEMM-like and reduction-like operators.

Installing with pre-built packages
==============================================================

You can install MIOpen on Ubuntu using ``apt-get install miopen-hip``.

If using OpenCL, you can use ``apt-get install miopen-opencl`` (but this is not recommended, as
OpenCL is deprecated).

Note that you can't install both backends on the same system simultaneously. If you want a different
backend other than what currently exists, completely uninstall the existing backend prior to installing
the new backend.

Installing with a kernels package
--------------------------------------------------------------------------------------------------------

MIOpen provides an optional pre-compiled kernels package to reduce startup latency. These
precompiled kernels comprise a select set of popular input configurations. We'll expand these kernels
in future releases to include additional coverage.

Note that all compiled kernels are locally cached in the ``$HOME/.cache/miopen/`` folder, so
precompiled kernels reduce the startup latency only for the first run of a neural network. Precompiled
kernels don't reduce startup time on subsequent runs.

To install the kernels package for your GPU architecture, use the following command:

.. code:: shell

  apt-get install miopen-hip-<arch>kdb

Where ``<arch>`` is the GPU architecture (e.g., ``gfx900``, ``gfx906``, ``gfx1030`` ).

.. note::

  Not installing these packages doesn't impact the functioning of MIOpen, since MIOpen compiles
  them on the target machine once you run the kernel. However, the compilation step may significantly
  increase the startup time for different operations.

The ``utils/install_precompiled_kernels.sh`` script provided as part of MIOpen automates the preceding
process. It queries the user machine for the GPU architecture and then installs the appropriate
package. You can invoke it using:

.. code:: shell

  ./utils/install_precompiled_kernels.sh

The preceding script depends on the ``rocminfo`` package to query the GPU architecture.

Installing dependencies
--------------------------------------------------------------------------------------------------------

You can install dependencies using the ``install_deps.cmake`` script (``cmake -P install_deps.cmake``).

By default, this installs to ``/usr/local``, but you can specify another location using the ``--prefix``
argument:

.. code:: shell

  cmake -P install_deps.cmake --prefix <miopen-dependency-path>

An example CMake step is:

.. code:: shell

  cmake -P install_deps.cmake --minimum --prefix /root/MIOpen/install_dir

You can use this prefix to specify the dependency path during the configuration phase using
``CMAKE_PREFIX_PATH``.

MIOpen's HIP backend uses :doc:`rocBLAS <rocblas:index>` by default. You can install rocBLAS'
minimum release using ``apt-get install rocblas``. To disable rocBLAS, set the configuration flag
``-DMIOPEN_USE_ROCBLAS=Off``. rocBLAS is **not** available with OpenCL.

MIOpen's HIP backend can use :doc:`hipBLASLt <hipblaslt:index>`. You can install hipBLASLt's minimum
release using ``apt-get install hipblaslt``. In addition to needing hipblaslt, you will also need to
install :doc:`hipBLAS <hipblas:index>`. You can install hipBLAS's minimum release using ``apt-get install hipblas``.
To disable hipBLASLt, set the configuration flag ``-DMIOPEN_USE_HIPBLASLT=Off``. 
hipBLASLt is **not** available with OpenCL.

Building MIOpen from source
================================================

You can build MIOpen form source with a HIP backend or an OpenCL backend.

HIP backend
--------------------------------------------------------------------------------------------------------

First, create a build directory:

.. code:: shell

  mkdir build; cd build;

Next, configure CMake. You can set the backend using the ``-DMIOPEN_BACKEND`` CMake variable.

Set the C++ compiler to ``clang++``. For the HIP backend (ROCm 3.5 and later), run:

.. code:: shell

  export CXX=<location-of-clang++-compiler>
  cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="<hip-installed-path>;<rocm-installed-path>;<miopen-dependency-path>" ..

An example CMake step is:

.. code:: shell

  export CXX=/opt/rocm/llvm/bin/clang++ && \
  cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..

.. note::

  When specifying the path for the `CMAKE_PREFIX_PATH` variable, **do not** use the tilde (`~`)
  shorthand to represent the home directory.

OpenCL backend
--------------------------------------------------------------------------------------------------------

.. note::

  OpenCL is deprecated. We recommend using a HIP backend and following the instructions listed in
  the preceding section.

First, run:

.. code:: shell

  cmake -DMIOPEN_BACKEND=OpenCL ..

The preceding code assumes OpenCL is installed in one of the standard locations. If not, then manually
set these CMake variables:

.. code:: shell

  cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=<hip-compiler-path> -DOPENCL_LIBRARIES=<opencl-library-path> -DOPENCL_INCLUDE_DIRS=<opencl-headers-path> ..

Here's an example dependency path for an environment in ROCm 3.5 and later:

.. code:: shell

  cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..

.. _setting-up-locations:

Setting up locations
--------------------------------------------------------------------------------------------------------

By default, the install location is set to ``/opt/rocm``. You can change this using
``CMAKE_INSTALL_PREFIX``:

.. code:: shell

  cmake -DMIOPEN_BACKEND=HIP -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..


System performance database and user database
--------------------------------------------------------------------------------------------------------

The default path to the system performance database (System PerfDb) is ``miopen/share/miopen/db/``
within the install location. The default path to the user performance database (User PerfDb) is
``~/.config/miopen/``. For development purposes, setting `BUILD_DEV` changes the default path to
both database files to the source directory:

.. code:: shell

  cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On ..


Database paths can be explicitly customized using the ``MIOPEN_SYSTEM_DB_PATH`` (System PerfDb)
and ``MIOPEN_USER_DB_PATH`` (User PerfDb) CMake variables.

To learn more, refer to the
:doc:`performance database <../conceptual/perfdb>` documentation.

Persistent program cache
--------------------------------------------------------------------------------------------------------

By default, MIOpen caches device programs in the ``~/.cache/miopen/`` directory. Within the cache
directory, there is a directory for each version of MIOpen. You can change the location of the cache
directory during configuration using the ``-DMIOPEN_CACHE_DIR=<cache-directory-path>`` flag.

You can also disable the cache during runtime using the ``MIOPEN_DISABLE_CACHE=1`` environmental
variable.

For MIOpen version 2.3 and earlier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the compiler changes, or you modify the kernels, then you must delete the cache for the MIOpen
version in use (e.g., ``rm -rf ~/.cache/miopen/<miopen-version-number>``). You can find more
information in the :doc:`cache <../conceptual/cache>` documentation.

For MIOpen version 2.4 and later
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MIOpen's kernel cache directory is versioned so that your cached kernels won't collide when upgrading
from an earlier version.

Changing the CMake configuration
--------------------------------------------------------------------------------------------------------

The configuration can be changed after running CMake (using ``ccmake``):

``ccmake ..`` **or** ``cmake-gui``: ``cmake-gui ..``

The ``ccmake`` program can be downloaded as a Linux package (``cmake-curses-gui``), but is not
available on Windows.

Building the library
=========================================================

You can build the library from the ``build`` directory using the 'Release' configuration:

``cmake --build . --config Release`` **or** ``make``

You can install it using the 'install' target:

``cmake --build . --config Release --target install`` **or** ``make install``

This installs the library to the ``CMAKE_INSTALL_PREFIX`` path that you specified.

Building the driver
=========================================================

MIOpen provides an application driver that you can use to run any layer in isolation, and measure
library performance and verification.

You can build the driver using the ``MIOpenDriver`` target:

``cmake --build . --config Release --target MIOpenDriver`` **or** ``make MIOpenDriver``

Running the tests
=========================================================

You can run tests using the 'check' target:

``cmake --build . --config Release --target check`` **or** ``make check``

To build and run a single test, use the following code:

.. code:: shell

  cmake --build . --config Release --target test_tensor
  ./bin/test_tensor

Formatting the code
=========================================================

All the code is formatted using `clang-format`. To format a file, use:

.. code:: shell

  clang-format-10 -style=file -i <path-to-source-file>

To format the code per commit, you can install githooks:

.. code:: shell

  ./.githooks/install

Storing large file using Git Large File Storage
=========================================================

Git Large File Storage (LFS) replaces large files, such as audio samples, videos, datasets, and graphics
with text pointers inside Git, while storing the file contents on a remote server. In MIOpen, we use Gi
LFS to store our large files, such as our kernel database files (``*.kdb``) that are normally > 0.5 GB.

You can install Git LFS using the following code:

.. code:: shell

  sudo apt install git-lfs
  git lfs install

In the Git repository where you want to use Git LFS, track the file type using the following code (if the
file type has already been tracked, you can skip this step):

.. code:: shell

  git lfs track "*.file_type"
  git add .gitattributes

You can pull all or a single large file using:

.. code:: shell

  git lfs pull --exclude=
  or
  git lfs pull --exclude= --include "filename"

Update the large files and push to GitHub using:

.. code:: shell

  git add my_large_files
  git commit -m "the message"
  git push

Installing the dependencies manually
===============================================================

If you're using Ubuntu v16, you can install the ``Boost`` packages using:

.. code:: shell

  sudo apt-get install libboost-dev
  sudo apt-get install libboost-system-dev
  sudo apt-get install libboost-filesystem-dev

.. note::

  By default, MIOpen attempts to build with Boost statically linked libraries. If required, you can build
  with dynamically linked Boost libraries using the `-DBoost_USE_STATIC_LIBS=Off` flag during the
  configuration stage. However, this is not recommended.

You must install the ``half`` header from the `half website <http://half.sourceforge.net/>`_.
