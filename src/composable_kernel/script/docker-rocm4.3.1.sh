WORKSPACE=$1
echo "workspace: " $WORKSPACE

docker run                                                                   \
-it                                                                          \
--rm                                                                         \
--privileged                                                                 \
--group-add sudo                                                             \
-w /root/workspace                                                           \
-v $WORKSPACE:/root/workspace                                                \
rocm/tensorflow:rocm4.3.1-tf2.6-dev                                          \
/bin/bash

#--network host                                                               \
