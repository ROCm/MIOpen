.. meta::
  :description: Build MIOpen using Docker
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
Build MIOpen using Docker
********************************************************************

You can build MIOpen using Docker by either downloading a prebuilt image or creating your own.

.. note::

    For ease of use, we recommended using the prebuilt Docker image.

* Downloading a prebuilt image

    You can find prebuilt Docker images at
    `ROCm Docker Hub <https://hub.docker.com/r/rocm/miopen/tags>`_.

* Building your own image

    .. code-block:: bash

        docker build -t miopen-image .

    To enter the development environment, use ``docker run``.  For example:

    .. code-block:: bash

        docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device /dev/dri:/dev/dri:rw
        --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker --group-add video
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined miopen-image

    Once in the Docker environment, use ``git clone MIOpen``. You can now start building MIOpen using
    CMake.

Building MIOpen from source
==========================================================

Use the following instructions to build MIOpen from source.

1. Create a build directory:

    .. code-block:: bash

        mkdir build; cd build;

2. Configure CMake using either an MIOpen or a HIP backend.

    **MIOpen backend**:

        Set your preferred backend using the ``-DMIOPEN_BACKEND`` CMake variable.

    **HIP backend (ROCm 3.5 and later)**:

        First, set the C++ compiler to ``clang++``:

            .. code-block:: bash

                export CXX=<location-of-clang++-compiler>

        Then, run the following command to configure CMake:

            .. code-block:: bash

                cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="<hip-installed-path>;<rocm-installed-path>;<miopen-dependency-path>" ..

        For example, you can set CMake to:

        .. code-block:: bash

            export CXX=/opt/rocm/llvm/bin/clang++ && \
            cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..

        .. note::

            When specifying the path for ``CMAKE_PREFIX_PATH``, **do not** use the tilde (~) shorthand
            for the home directory.


Choosing an install location
-------------------------------------------------------------------------------------

By default, the install location is set to ``/opt/rocm``. If you used a different install location, set your
install path using ``CMAKE_INSTALL_PREFIX``:

.. code-block:: bash

    cmake -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..
