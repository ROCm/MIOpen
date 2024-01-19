.. _install:

Building MIOpen from source
~~~~~~~~~~

Configuring with cmake
----------

First create a build directory:

.. code-block:: bash

    mkdir build; cd build;


Next configure cmake. The preferred backend for MIOpen can be set using the `-DMIOPEN_BACKEND` cmake variable.

For the HIP backend (ROCm 3.5 and later), run
----------

Set the C++ compiler to `clang++`.

.. code-block:: bash

    export CXX=<location-of-clang++-compiler>
    cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="<hip-installed-path>;<rocm-installed-path>;<miopen-dependency-path>" ..

An example cmake step can be:

.. code-block:: bash

    export CXX=/opt/rocm/llvm/bin/clang++ && \
    cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..

Note: When specifying the path for the `CMAKE_PREFIX_PATH` variable, **do not** use the `~` shorthand for the user home directory.


Setting Up Locations
----------

By default the install location is set to '/opt/rocm', this can be set by using `CMAKE_INSTALL_PREFIX`:

.. code-block:: bash

    cmake -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..

Building MIOpen using docker
~~~~~~~~~~

The easiest way is to use docker. You can build the top-level docker file:

.. code-block:: bash

    docker build -t miopen-image .

Then to enter the development environment use `docker run`, for example:

.. code-block:: bash

    docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device /dev/dri:/dev/dri:rw  --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined miopen-image

Prebuilt docker images can be found on [ROCm's public docker hub here](https://hub.docker.com/r/rocm/miopen/tags).