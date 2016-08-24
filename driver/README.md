# MLOpen Driver

The driver enables to test forward/backward convolution in MLOpen.
To build the driver, type `make MLOpenDriver` from the `build` directory.

To execute from the build directory: `./driver/MLOpenDriver`

Sample runs:
* With search on - `./driver/MLOpenDriver -c 32 -H 8 -W 8 -k 64 -x 5 -y 5 -p 1 -q 1` (runs forward & backward)
* With search off - `./driver/MLOpenDriver -c 32 -H 8 -W 8 -k 64 -x 5 -y 5 -s 0 -F 1 -p 1 -q 1` (runs only forward)
```
      --forwconv           -F        Run only Forward Convolution (Default=0)
      --in_h               -H        Input Height (Default=32)
      --printconv          -P        Print Convolution Dimensions (Default=1)
      --verify             -V        Verify Each Layer (Default=1)
      --in_w               -W        Input Width (Default=32)
      --in_channels        -c        Number of Input Channels (Default=3)
      --help               -h        Print Help Message
      --iter               -i        Number of Iterations (Default=10)
      --out_channels       -k        Number of Output Channels (Default=32)
      --batchsize          -n        Mini-batch size (Default=100)
      --pad_h              -p        Zero Padding Height (Default=0)
      --pad_w              -q        Zero Padding Width (Default=0)
      --pad_val            -r        Padding Value (Default=0)
      --search             -s        Search Kernel Config (Default=0)
      --time               -t        Time Each Layer (Default=0)
      --conv_stride_0      -u        Convolution Stride Vertical (Default=1)
      --conv_stride_1      -v        Convolution Stride Horizontal (Default=1)
      --fil_h              -x        Filter Height (Default=3)
      --fil_w              -y        Filter Width (Default=3)
```
