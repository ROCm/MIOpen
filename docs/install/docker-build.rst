.. meta::
  :description: Build MIOpen using Docker
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
Build MIOpen using Docker
********************************************************************

You can build MIOpen using Docker in the following ways,

-	Download prebuilt Docker
-	Build your own Docker

Note: It is recommended that you build MIOpen using the prebuilt Docker, as it is easily and publicly available. 

Download the prebuilt Docker
-----------------------------------
You can find prebuilt Docker images at the public `ROCm Docker hub. <https://hub.docker.com/r/rocm/miopen/tags>`_

Build your own Docker
----------------------

Build the top-level Docker file using the following instructions, 

.. code-block:: bash

    docker build -t miopen-image .


Note: To enter the development environment, use *docker run*.  For example,

.. code-block:: bash

    docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device /dev/dri:/dev/dri:rw  --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined miopen-image

You can now enter the Docker environment, git clone MIOpen, and start building MIOpen using CMake. 


Build MIOpen from source
~~~~~~~~~~

Use the following instructions to build MIOpen from source.

Configure with CMake
----------

1. Use the command below to create a build directory:

.. code-block:: bash

    mkdir build; cd build;


2. Configure CMake. 

MIOpen backend
*****************
The preferred backend for MIOpen can be set using the `-DMIOPEN_BACKEND` CMake variable. For more details, refer to `Build MIOpen from source <https://github.com/ROCm/MIOpen?tab=readme-ov-file#building-miopen-from-source>`_

HIP backend (ROCm 3.5 and later)
********************************
For the HIP backend , 

1. Set the C++ compiler to `clang++`.

.. code-block:: bash

    export CXX=<location-of-clang++-compiler>

2. Run the following command to configure CMake.

.. code-block:: bash

    cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="<hip-installed-path>;<rocm-installed-path>;<miopen-dependency-path>" ..

For example, you can set CMake to,

.. code-block:: bash

    export CXX=/opt/rocm/llvm/bin/clang++ && \
    cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..

Note: When specifying the path for the `CMAKE_PREFIX_PATH` variable, **do not** use the `~` shorthand for the user Home directory.


Setting up locations to install MIOpen
----------

By default, the install location is set to '/opt/rocm'. You can use the following instruction to set the install location using `CMAKE_INSTALL_PREFIX`,


.. code-block:: bash

    cmake -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..

