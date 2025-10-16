.. meta::
  :description: Building MIOpen from source
  :keywords: MIOpen, ROCm, API, documentation, build, installation, testing

********************************************************************
Build MIOpen from source
********************************************************************

This topic discusses how to build MIOpen from source and configure the resulting application.
It also explains how to build the library and driver and run the tests. For a list of MIOpen
prerequisites, see :doc:`MIOpen prerequisites <./prerequisites>`. To install MIOpen from a
package, see :doc:`install MIOpen <./install>`.

Building MIOpen
================================================

You can build MIOpen form source using either a HIP backend or an OpenCL backend.

HIP backend
--------------------------------------------------------------------------------------------------------

To build MIOpen using the HIP backend (in ROCm 3.5 and later), follow these steps:

#. Create the build directory:

   .. code:: shell

      mkdir build; cd build;

#. Configure CMake. Set the backend using the ``-DMIOPEN_BACKEND`` CMake variable and
   set the C++ compiler to ``clang++``. The command to build MIOpen with a HIP backend follows this format:

   .. code:: shell

      export CXX=<location-of-clang++-compiler>
      cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="<hip-installed-path>;<rocm-installed-path>;<miopen-dependency-path>" ..

   An example of a CMake build is:

   .. code:: shell

      export CXX=/opt/rocm/llvm/bin/clang++ && \
      cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..

   .. note::

      When specifying the path for the ``CMAKE_PREFIX_PATH`` variable, **do not** use the tilde (``~``)
      symbol to represent the home directory.

OpenCL backend
--------------------------------------------------------------------------------------------------------

To build MIOpen using an OpenCL backend, run the following command:

.. code:: shell

   cmake -DMIOPEN_BACKEND=OpenCL ..

.. note::

   OpenCL is deprecated and the HIP backend is recommended instead. To install MIOpen using HIP, follow the instructions in
   the preceding section.

The preceding code assumes OpenCL is installed in one of the standard locations. If not, then manually
set these CMake variables:

.. code:: shell

   cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=<hip-compiler-path> -DOPENCL_LIBRARIES=<opencl-library-path> -DOPENCL_INCLUDE_DIRS=<opencl-headers-path> ..

Here's an example showing how to configure the dependency path for an environment (applies to ROCm version 3.5 and later):

.. code:: shell

   cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..

.. _setting-up-locations:

Setting the install location
--------------------------------------------------------------------------------------------------------

By default, the install location is set to ``/opt/rocm``. To change this, use ``CMAKE_INSTALL_PREFIX``:

.. code:: shell

   cmake -DMIOPEN_BACKEND=HIP -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..

System performance database and user database
--------------------------------------------------------------------------------------------------------

The default path to the system performance database (System PerfDb) is ``miopen/share/miopen/db/``
within the install location. The default path to the user performance database (User PerfDb) is
``~/.config/miopen/``. Setting ``BUILD_DEV`` for development purposes changes the default path for
both database files to the source directory:

.. code:: shell

  cmake -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On ..

To customize the database paths, use the ``MIOPEN_SYSTEM_DB_PATH`` (for the System PerfDb)
and ``MIOPEN_USER_DB_PATH`` (for the User PerfDb) CMake variables.

To learn more, see :doc:`using the performance database <../conceptual/perfdb>`.

Persistent program cache
--------------------------------------------------------------------------------------------------------

By default, MIOpen caches device programs in the ``~/.cache/miopen/`` directory. Within the cache
directory, there is a directory for each version of MIOpen. To change the location of the cache
directory during configuration, use the ``-DMIOPEN_CACHE_DIR=<cache-directory-path>`` flag.

To disable the cache during runtime, set the ``MIOPEN_DISABLE_CACHE=1`` environmental
variable.

For MIOpen version 2.3 and earlier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the compiler changes or you modify the kernels, then you must delete the cache for the MIOpen
version in use, for example, ``rm -rf ~/.cache/miopen/<miopen-version-number>``. For more
information, see :doc:`kernel cache <../conceptual/cache>`.

For MIOpen version 2.4 and later
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MIOpen's kernel cache directory is versioned, so cached kernels don't collide when upgrading
from an earlier version.

Changing the CMake configuration
--------------------------------------------------------------------------------------------------------

The configuration can be changed after running CMake by using ``ccmake``:

.. code:: shell

   ccmake ..

or

.. code:: shell

   cmake-gui: cmake-gui ..

The ``ccmake`` program can be downloaded using the Linux package ``cmake-curses-gui``, but is not
available on Windows.

Building the library
=========================================================

You can build the library from the ``build`` directory using the ``Release`` configuration:

.. code:: shell

   cmake --build . --config Release

or

.. code:: shell

   make

You can install it using the ``install`` target:

.. code:: shell

   cmake --build . --config Release --target install

or

.. code:: shell

   make install

This command installs the library to the ``CMAKE_INSTALL_PREFIX`` directory path.

Building the driver
=========================================================

MIOpen provides an application driver that can run any layer in isolation and measure
library performance and verification.

To build the driver, use the ``MIOpenDriver`` target:

.. code:: shell

   cmake --build . --config Release --target MIOpenDriver

or

.. code:: shell

   make MIOpenDriver

Running the tests
=========================================================

To run tests, use the ``check`` target:

.. code:: shell

   cmake --build . --config Release --target check

or

.. code:: shell

   make check

To build and run a single test, use the following commands:

.. code:: shell

   cmake --build . --config Release --target test_tensor
   ./bin/test_tensor

Formatting the code
=========================================================

To format the code using ``clang-format``, use this command:

.. code:: shell

  clang-format -style=file -i <path-to-source-file>

To format the code per commit, install githooks:

.. code:: shell

  ./.githooks/install

Storing large file using Git Large File Storage
=========================================================

`Data Versioning System (DVS) <https://dvc.org/>`_ replaces large files, such as audio samples, videos, datasets, and 
graphics with text pointers inside Git, while storing the file contents on a remote server. MIOpen uses DVC to 
store large files, such as kernel database files (``*.kdb``), which are normally > 0.5 GB.

To install DVC, use the `instructions provided for your platform here <https://dvc.org/doc/install>`_.

You can `pull <https://dvc.org/doc/command-reference/pull>`_ all large files or a single large file using:

.. code:: shell

   dvc pull

or

.. code:: shell

   dvc pull "filename"


If you are familiar with using Git LFS, a key difference with DVC is that you must manually run ``dvc pull`` after you 
switch branches or merge changes in Git to ensure any large binaries are kept in sync with your checkout.

Installing the dependencies manually
===============================================================

If you're using Ubuntu v16, you can install the ``Boost`` packages using:

.. code:: shell

   sudo apt-get install libboost-dev
   sudo apt-get install libboost-system-dev
   sudo apt-get install libboost-filesystem-dev

.. note::

   By default, MIOpen attempts to build with Boost statically linked libraries. To build
   with dynamically linked Boost libraries, use the ``-DBoost_USE_STATIC_LIBS=Off`` flag during the
   configuration stage. However, this is not recommended.

You must install the ``half`` header from the `half website <http://half.sourceforge.net/>`_.
