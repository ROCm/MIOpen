.. meta::
  :description: Installing MIOpen from a package
  :keywords: MIOpen, ROCm, API, documentation, install, package

********************************************************************
Install MIOpen
********************************************************************

To install MIOpen from a package, choose either a pre-built package for your Linux
distribution or choose a pre-compiled kernels package. For a list of MIOpen
prerequisites, see :doc:`MIOpen prerequisites <./prerequisites>`. To build MIOpen from
source, see :doc:`build MIOpen from source <./build-source>`.

Installing using a pre-built package
==============================================================

To install MIOpen on Ubuntu, use ``apt-get install miopen-hip``.

If you are using OpenCL, use ``apt-get install miopen-opencl``. (This is not recommended because
OpenCL is deprecated.)

.. note::

   You can't install both backends on the same system simultaneously. To switch to a different
   backend, completely uninstall the existing backend prior to installing
   the new backend.

Installing using a kernels package
==============================================================

MIOpen provides an optional pre-compiled kernels package to reduce startup latency. These
precompiled kernels consist of a select set of popular input configurations. This collection of kernels
will continue to expand to include additional coverage.

.. note::

   All compiled kernels are locally cached in the ``$HOME/.cache/miopen/`` folder, so these
   pre-compiled kernels only reduce the startup latency for the first run of a neural network. Pre-compiled
   kernels don't reduce the startup time on subsequent runs.

To install the kernels package for your GPU architecture, use the following command:

.. code:: shell

   apt-get install miopen-hip-<arch>kdb

Where ``<arch>`` is the GPU architecture, for example, ``gfx900``, ``gfx906``, or ``gfx1030``.

.. note::

   If you don't install these packages, it doesn't impact the functioning of MIOpen. This is because MIOpen compiles
   them on the target machine after you run the kernel. However, the compilation step might significantly
   increase the startup time for certain operations.

The ``utils/install_precompiled_kernels.sh`` script provided as part of MIOpen automates the preceding
process. It queries the user machine for the GPU architecture and then installs the appropriate
package. To run it, use the following command:

.. code:: shell

   ./utils/install_precompiled_kernels.sh

The preceding script depends on the ``rocminfo`` package to query the GPU architecture.

Installing dependencies
==============================================================

To install the MIOpen dependencies, use the ``install_deps.cmake`` command:

.. note::

   You can run ``install_deps.cmake`` from the ``rocm-libraries/projects/miopen`` directory.


.. code:: shell

   cmake -P install_deps.cmake

By default, this installs the dependencies in ``/usr/local``, but you can specify another location using the ``--prefix``
argument:

.. code:: shell

  cmake -P install_deps.cmake --prefix <miopen-dependency-path>

The following example demonstrates how to use ``cmake`` with a specific installation directory:

.. code:: shell

   cmake -P install_deps.cmake --minimum --prefix /root/MIOpen/install_dir

You can specify this directory during the configuration phase using ``CMAKE_PREFIX_PATH``.

MIOpen's HIP backend uses :doc:`rocBLAS <rocblas:index>` by default. You can install the rocBLAS
minimum release using ``apt-get install rocblas``. To disable rocBLAS, set the configuration flag
``-DMIOPEN_USE_ROCBLAS=Off``. rocBLAS is **not** available with OpenCL.

MIOpen's HIP backend can use :doc:`hipBLASLt <hipblaslt:index>`. To install the minimum release of hipBLASLt,
use ``apt-get install hipblaslt``. In addition to installing hipBLASLt, you must also
install :doc:`hipBLAS <hipblas:index>`. To install the hipBLAS minimum release, use ``apt-get install hipblas``.
To disable hipBLASLt, set the configuration flag ``-DMIOPEN_USE_HIPBLASLT=Off``.
hipBLASLt is **not** available with OpenCL.
