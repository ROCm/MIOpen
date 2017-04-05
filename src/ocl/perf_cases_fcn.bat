set B=%1
set S=%2
set V=%3
set T=%4
cd ../../build
PATH=.\src\Debug;%PATH%
echo "FullConvNet"
:fullconvnet
rem goto 4a
echo "conv1/7x7_s2 1x3x1024x2048x64x7x7"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  2048 -H 1024 -c 3 -k 64 -x 7 -y 7 -p 3 -q 3  -F 1 -u 2 -v 2
echo "conv1/relu_7x7 1x64x512x1024x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  1024 -H 512 -c 64 -F 1
echo "pool1/3x3_s2 1x64x512x1024x64x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  1024 -H 512 -c 64 -k 64 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1
echo "pool1/norm 1x64x256x512x64x5x5"
.\driver\Debug\MIOpenDriver.exe lrn -i 1 -t %T% -n %B%  -V %V% -W  512 -H 256 -c 64 -F 1
echo "conv2/3x3_reduce 1x64x256x512x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  512 -H 256 -c 64 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "conv2/relu_3x3_reduce 1x64x256x512x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  512 -H 256 -c 64 -F 1
echo "conv2/3x3 1x64x256x512x192x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  512 -H 256 -c 64 -k 192 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "conv2/relu_3x3 1x192x256x512x192"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  512 -H 256 -c 192 -F 1
echo "conv2/norm2 1x192x256x512x192x5x5"
.\driver\Debug\MIOpenDriver.exe lrn -i 1 -t %T% -n %B%  -V %V% -W  512 -H 256 -c 192 -F 1
echo "pool2/3x3_s2 1x192x256x512x192x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  512 -H 256 -c 192 -k 192 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1
echo "inception_3a/1x1 1x192x128x256x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 192 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_3a/relu_1x1 1x64x128x256x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 64 -F 1
echo "inception_3a/3x3_reduce 1x192x128x256x96x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 192 -k 96 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_3a/relu_3x3_reduce 1x96x128x256x96"
.\driver\Debug\MIOpenDriver.exe lrn -t %T% -n %B%  -V %V% -W  256 -H 128 -c 96 -k 96 -F 1
echo "inception_3a/3x3 1x96x128x256x128x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 96 -k 128 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_3a/relu_3x3 1x128x128x256x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 128 -F 1
echo "inception_3a/5x5_reduce 1x192x128x256x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 192 -k 16 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_3a/relu_5x5_reduce 1x16x128x256x16"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 16 -F 1
echo "inception_3a/5x5 1x16x128x256x32x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 16 -k 32 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_3a/relu_5x5 1x32x128x256x32"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 32 -F 1
echo "inception_3a/pool 1x192x128x256x192x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  256 -H 128 -c 192 -k 192 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_3a/pool_proj 1x192x128x256x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 192 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_3a/relu_pool_proj 1x32x128x256x32"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 32 -F 1
echo "inception_3a/output 1x128x256x256"
:3b
echo "inception_3b/1x1 1x256x128x256x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 256 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "nception_3b/relu_1x1 1x128x128x256x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 128 -F 1
echo "inception_3b/3x3_reduce 1x256x128x256x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 256 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_3a/relu_3x3_reduce 1x128x128x256x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 128 -F 1
echo "inception_3b/3x3 1x128x128x256x192x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 128 -k 192 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_3b/relu_3x3 1x192x128x256x192"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 192 -F 1
echo "inception_3b/5x5_reduce 1x256x128x256x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 256 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_3b/relu_5x5_reduce 1x32x128x256x32"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 32 -F 1
echo "inception_3b/5x5 1x32x128x256x96x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 32 -k 96 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_3b/relu_5x5 1x96x128x256x96"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 96 -F 1
echo "inception_3b/pool 1x256x128x256x256x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  256 -H 128 -c 256 -k 256 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_3b/pool_proj 1x256x128x256x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  256 -H 128 -c 256 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_3b/relu_pool_proj 1x64x128x256x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  256 -H 128 -c 64 -F 1
echo "inception_3b/output 3"
echo "pool3/3x3_s2 1x480x128x256x480x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  256 -H 128 -c 480 -k 480 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1
:4a
echo "inception_4a/1x1 1x480x64x128x192x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 480 -k 192 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "nception_4a/relu_1x1 1x192x64x128x192x3x3"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 192 -F 1
echo "inception_4a/3x3_reduce 1x480x64x128x96x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 480 -k 96 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4a/relu_3x3_reduce 1x96x64x128x96"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 96 -F 1
echo "inception_4a/3x3 1x96x64x128x208x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 96 -k 208 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_4a/relu_3x3 1x208x64x128x208"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 208 -F 1
echo "inception_4a/5x5_reduce 1x480x64x128x16x1s1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 480 -k 16 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4a/relu_5x5_reduce 1x16x64x128x16"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 16 -F 1
echo "inception_4a/5x5 1x16x64x128x48x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 16 -k 48 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_4a/relu_5x5_reduce 1x48x64x128x48"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 48 -F 1
echo "inception_4a/pool 32x480x64x128x480x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  128 -H 64 -c 480 -k 480 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_4a/pool_proj 1x480x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 480 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4a/relu_pool_proj 1x64x64x128x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 64 -F 1
echo "inception_4a/output 3"
echo "inception_4b/1x1 1x512x64x128x160x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 160 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4b/relu_1x1 1x160x64x128x160"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 160 -F 1
echo "inception_4b/3x3_reduce 1x512x64x128x112x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 112 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4b/relu_3x3_reduce 1x112x64x128x112"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 112 -F 1
echo "inception_4b/3x3 1x112x64x128x224x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 112 -k 224 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_4b/relu_3x3 1x224x64x128x224"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 224 -F 1
echo "inception_4b/5x5_reduce 1x512x64x128x24x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 128 -c 512 -k 24 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4b/relu_5x5_reduce 1x24x64x128x24"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 24 -F 1
echo "inception_4b/5x5 1x24x64x128x64x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 24 -k 64 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_4b/relu_5x5 1x64x64x128x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 64 -F 1
echo "inception_4b/pool 1x512x64x128x512x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 512 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_4b/pool_proj 1x512x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4b/relu_pool_proj 1x64x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 64 -F 1
echo "inception_4b/output 3"
echo "inception_4c/1x1 1x512x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4c/relu_1x1 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_4c/3x3_reduce 1x512x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4c/relu_3x3_reduce 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_4c/3x3 1x128x64x128x256x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -k 256 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_4c/relu_3x3 1x256x64x128x256"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 256 -F 1
echo "inception_4c/5x5_reduce 1x512x64x128x24x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 24 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4c/relu_5x5_reduce 1x24x64x128x24"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 24 -F 1
echo "inception_4c/5x5 1x24x64x128x64x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 24 -k 64 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_4c/relu_5x5 1x64x64x128x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 64 -F 1
echo "inception_4c/pool 1x512x64x128x512x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 512 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_4c/pool_proj 1x512x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4c/relu_pool_proj 1x64x64x128x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 64 -F 1
echo "inception_4c/output 3"
echo "inception_4d/1x1 1x512x64x128x112x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 112 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4d/relu_1x1 1x112x64x128x112"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 112 -F 1
echo "inception_4d/3x3_reduce 1x512x64x128x144x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 144 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4d/relu_3x3_reduce 1x144x64x128x144"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 144 -F 1
echo "inception_4d/3x3 1x144x64x128x288x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 144 -k 288 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_4d/relu_3x3 1x288x64x128x288"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 288 -F 1
echo "inception_4d/5x5_reduce 1x512x64x128x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4d/relu_5x5_reduce 1x32x64x128x32"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 32 -F 1
echo "inception_4d/5x5 1x32x64x128x64x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 32 -k 64 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_4d/relu_5x5 1x64x64x128x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 64 -F 1
echo "inception_4d/pool 1x512x64x128x512x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 512 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_4d/pool_proj 1x512x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 512 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4d/relu_pool_proj 1x64x64x128x64"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 64 -F 1
echo "inception_4d/output 3"
echo "inception_4e/1x1 1x528x64x128x256x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 528 -k 256 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4e/relu_1x1 1x256x64x128x256"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 256 -F 1
echo "inception_4e/3x3_reduce 1x528x64x128x160x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 528 -k 160 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4e/relu_3x3_reduce 1x160x64x128x160"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 160 -F 1
echo "inception_4e/3x3 1x160x64x128x320x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 160 -k 320 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_4e/relu_3x3 1x320x64x128x320"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 320 -F 1
echo "inception_4e/5x5_reduce 1x528x64x128x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 528 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4e/relu_5x5_reduce 1x32x64x128x32"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 32 -F 1
echo "inception_4e/5x5 1x32x64x128x128x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 32 -k 128 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_4e/relu_5x5 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_4e/pool 1x528x64x128x528x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  128 -H 64 -c 528 -k 528 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_4e/pool_proj 1x528x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 528 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_4e/relu_pool_proj 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_4e/output 3"
rem goto end
echo "inception_5a/1x1 1x832x64x128x256x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 256 -x 1 -y 1 -p 0 -q 0  -F 1 -s 1
echo "inception_5a/relu_1x1 1x256x64x128x256"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 256 -F 1
echo "inception_5a/3x3_reduce 1x832x64x128x160x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 160 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_5a/relu_3x3_reduce 1x160x64x128x160"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 160 -F 1
echo "inception_5a/3x3 1x160x64x128x320x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 160 -k 320 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_5a/relu_3x3 1x320x64x128x320"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 320 -F 1
echo "inception_5a/5x5_reduce 1x832x64x128x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_5a/relu_5x5_reduce 1x32x64x128x32"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 32 -F 1
echo "inception_5a/5x5 1x32x64x128x128x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 32 -k 128 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_5a/relu_5x5 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_5a/pool 1x832x64x128x832x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 832 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_5a/pool_proj 1x832x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_5a/relu_pool_proj 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_5a/output 3"
echo "inception_5b/1x1 1x832x64x128x384x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 384 -x 1 -y 1 -p 0 -q 0  -F 1 -s 1
echo "inception_5b/relu_1x1 1x384x64x384x256"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 384 -F 1
echo "inception_5b/3x3_reduce 1x832x64x128x192x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 192 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_5b/relu_3x3_reduce 1x192x64x128x192"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 192 -F 1
echo "inception_5b/3x3 1x192x64x128x384x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 192 -k 384 -x 3 -y 3 -p 1 -q 1  -F 1 -s %S%
echo "inception_5b/relu_3x3 1x384x64x128x384"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 384 -F 1
echo "inception_5b/5x5_reduce 1x832x64x128x48x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 48 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_5b/relu_5x5_reduce 1x48x64x128x48"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 48 -F 1
echo "inception_5b/5x5 1x48x64x128x128x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 48 -k 128 -x 5 -y 5 -p 2 -q 2  -F 1 -s %S%
echo "inception_5b/relu_5x5 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_5b/pool 1x832x64x128x832x3x3"
.\driver\Debug\MIOpenDriver.exe pool -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 832 -x 3 -y 3 -p 1 -q 1 -F 1
echo "inception_5b/pool_proj 1x832x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -t %T% -n %B%  -V %V% -W  128 -H 64 -c 832 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -s %S%
echo "inception_5b/relu_pool_proj 1x128x64x128x128"
.\driver\Debug\MIOpenDriver.exe activ -t %T% -n %B%  -V %V% -W  128 -H 64 -c 128 -F 1
echo "inception_5b/output 3"
:end
cd ../src/OCL