#!/bin/bash

## GPU visibility
 export ROCR_VISIBLE_DEVICE=0
 export GPU_DEVICE_ORDINAL=0

## Boost
 export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

## Compiling
#export OLC_DEBUG_HIP_VERBOSE=1
#export OLC_DEBUG_HIP_DUMP=1
#export OLC_DEBUG_SAVE_TEMP_DIR=1

 make -j conv_fwd_driver_offline
 make -j conv_bwd_driver_offline
 make -j conv_fwd_driver_online

#rm -rf /root/_hip_binary_kernels_/
#rm -rf /tmp/olCompile*

LAYOUT=$1
ALGO=$2
VERIFY=$3
INIT=$4
LOG=$5
REPEAT=$6

################################################ layout  algo  verify  init  log  repeat  N__ K___ C___ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  128  192 3 3  71   71     2 2       1 1      1 1       1 1
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256 1024 1 7  17   17     1 1       1 1      0 3       0 3
 ./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  14   14     1 1       1 1      1 1       1 1
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  128  128 3 3  14   14     1 1       1 1      1 1       1 1
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3   7    7     1 1       1 1      1 1       1 1

#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  512  192 3 3  35   35     2 2       1 1      0 0       0 0
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  30   30     2 2       1 1      0 0       0 0
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3  16   16     2 2       1 1      0 0       0 0

#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256 2048 1024 1 1  14   14     2 2       1 1      0 0       0 0
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256 1024 1 1  14   14     1 1       1 1      0 0       0 0
#./host/driver_offline/conv_fwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512 2048 1 1   7    7     1 1       1 1      0 0       0 0

#./host/driver_offline/conv_bwd_driver_offline  $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  14   14     1 1       1 1      1 1       1 1

#./host/driver_online/conv_fwd_driver_online    $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
