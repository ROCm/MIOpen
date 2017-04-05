set B=%1
set S=%2
set V=%3
set T=%4
cd ../../build
PATH=.\src\Debug;%PATH%
echo "GoogleNet"
echo "conv1/7x7_s2 10x3x224x224x64x7x7"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 224 -H 224 -c 3 -k 64 -x 7 -y 7 -p 3 -q 3  -F 1 -t %T% -V %V% -u 2 -v 2
echo "conv1/relu_7x7 1x64x112x112x64"
rem .\driver\Debug\MIOpenDriver.exe active -i 1 -n %B% -W 112 -H 112 -c 64 -F 1 -t %T% -V %V%
echo "pool1/3x3_s2 10x64x112x112x64x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 112 -H 112 -c 64 -k 64 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1 -t %T% -V %V%
echo "pool1/norm 1x64x56x56x64x5x5"
.\driver\Debug\MIOpenDriver.exe lrn -i 1 -n %B% -W 56 -H 56 -c 64 -F 1 -t %T% -V %V%
echo "conv2/3x3_reduce 10x64x56x56x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 56 -H 56 -c 64 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "conv2/relu_3x3_reduce 1x64x256x512x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 56 -H 56 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "conv2/3x3 10x64x56x56x192x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 56 -H 56 -c 64 -k 192 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "conv2/relu_3x3 1x192x256x512x192"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 56 -H 56 -c 192 -k 192 -F 1 -t %T% -V %V%
echo "conv2/norm2 1x192x256x512x192x5x5"
.\driver\Debug\MIOpenDriver.exe lrn -i 1 -n %B% -W 56 -H 56 -c 192 -F 1 -t %T% -V %V%
echo "pool2/3x3_s2 10x192x56x56x192x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 56 -H 56 -c 192 -k 192 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1 -t %T% -V %V%
echo "inception_3a/1x1 1x192x28x28x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 192 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_3a/3x3_reduce 1x192x128x256x96x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 192 -k 96 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_3a/relu_3x3_reduce 1x96x128x256x96"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 96 -k 96 -F 1 -t %T% -V %V%
echo "inception_3a/3x3 10x96x28x28x128x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 96 -k 128 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_3a/relu_3x3 1x128x128x256x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_3a/5x5_reduce 1x192x128x256x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 192 -k 16 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_3a/relu_5x5_reduce 1x16x128x256x16"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 16 -k 16 -F 1 -t %T% -V %V%
echo "inception_3a/5x5 10x16x28x28x32x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 16 -k 32 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_3a/relu_5x5 1x32x128x256x32"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 32 -k 32 -F 1 -t %T% -V %V%
echo "inception_3a/pool 10x192x28x28x192x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 28 -H 28 -c 192 -k 192 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_3a/pool_proj 1x192x128x256x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 192 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_3a/relu_pool_proj 1x32x128x256x32"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 32 -k 32 -F 1 -t %T% -V %V%
echo "inception_3a/output 1x128x256x256"
rem goto end
:3b
echo "inception_3b/1x1 1x256x128x256x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 256 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "nception_3b/relu_1x1 1x128x128x256x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_3b/3x3_reduce 1x256x128x256x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 256 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_3a/relu_3x3_reduce 1x128x128x256x128"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_3b/3x3 10x128x28x28x192x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 128 -k 192 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_3b/relu_3x3 1x192x128x256x192"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 192 -k 192 -F 1 -t %T% -V %V%
echo "inception_3b/5x5_reduce 1x256x128x256x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 256 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_3b/relu_5x5_reduce 1x32x128x256x32"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 32 -k 32 -F 1 -t %T% -V %V%
echo "inception_3b/5x5 10x32x28x28x96x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 32 -k 96 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_3b/relu_5x5 1x96x128x256x96"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 96 -k 96 -F 1 -t %T% -V %V%
echo "inception_3b/pool 10x256x28x28x256x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 28 -H 28 -c 256 -k 256 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_3b/pool_proj 1x256x128x256x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 256 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_3b/relu_pool_proj 1x64x128x256x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 28 -H 28 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_3b/output 3"
echo "pool3/3x3_s2 1x480x28x28x480x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 28 -H 28 -c 480 -k 480 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1 -t %T% -V %V%
rem goto end
:4a
echo "inception_4a/1x1 1x480x64x128x192x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 480 -k 192 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "nception_4a/relu_1x1 1x192x64x128x192x3x3"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 192 -k 192 -F 1 -t %T% -V %V%
"inception_4a/3x3_reduce 1x480x64x128x96x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 480 -k 96 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4a/relu_3x3_reduce 1x96x64x128x96"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 96 -k 96 -F 1 -t %T% -V %V%
echo "inception_4a/3x3 10x96x14x14x208x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 96 -k 208 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_4a/relu_3x3 1x208x64x128x208"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 208 -k 208 -F 1 -t %T% -V %V%
echo "inception_4a/5x5_reduce 1x480x64x128x16x1s1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 480 -k 16 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4a/relu_5x5_reduce 1x16x64x128x16"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 16 -k 16 -F 1 -t %T% -V %V%
echo "inception_4a/5x5 10x16x14x14x48x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 16 -k 48 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_4a/relu_5x5_reduce 1x48x64x128x48"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 48 -k 48 -F 1 -t %T% -V %V%
echo "inception_4a/pool 10x480x14x14x480x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 14 -H 14 -c 480 -k 480 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_4a/pool_proj 1x480x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 480 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4a/relu_pool_proj 1x64x64x128x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_4a/output 3"
rem goto end
:4b
echo "inception_4b/1x1 1x512x64x128x160x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 160 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4b/relu_1x1 1x160x64x128x160"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 160 -k 160 -F 1 -t %T% -V %V%
"inception_4b/3x3_reduce 1x512x64x128x112x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 112 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4b/relu_3x3_reduce 1x112x64x128x112"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 112 -k 112 -F 1 -t %T% -V %V%
echo "inception_4b/3x3 10x112x14x14x224x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 112 -k 224 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_4b/relu_3x3 1x224x64x128x224"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 224 -k 224 -F 1 -t %T% -V %V%
"inception_4b/5x5_reduce 1x512x64x128x24x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 24 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4b/relu_5x5_reduce 1x24x64x128x24"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 24 -k 24 -F 1 -t %T% -V %V%
echo "inception_4b/5x5 10x24x14x14x64x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 24 -k 64 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_4b/relu_5x5 1x64x64x128x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_4b/pool 10x512x14x14x512x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 14 -H 14 -c 512 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_4b/pool_proj 1x512x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4b/relu_pool_proj 1x64x64x128x64x1x1"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_4b/output 3"
echo "inception_4c/1x1 1x512x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4c/relu_1x1 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_4c/3x3_reduce 1x512x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4c/relu_3x3_reduce 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_4c/3x3 10x128x14x14x256x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 128 -k 256 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_4c/relu_3x3 1x256x64x128x256"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 256 -k 256 -F 1 -t %T% -V %V%
echo "inception_4c/5x5_reduce 1x512x64x128x24x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 24 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4c/relu_5x5_reduce 1x24x64x128x24"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 24 -k 24 -F 1 -t %T% -V %V%
echo "inception_4c/5x5 10x24x14x14x64x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 24 -k 64 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_4c/relu_5x5 1x64x64x128x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_4c/pool 10x512x14x14x512x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 14 -H 14 -c 512 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_4c/pool_proj 1x512x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4c/relu_pool_proj 1x64x64x128x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_4c/output 3"
echo "inception_4d/1x1 1x512x64x128x112x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 112 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4d/relu_1x1 1x112x64x128x112"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 112 -k 112 -F 1 -t %T% -V %V%
echo "inception_4d/3x3_reduce 1x512x64x128x144x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 144 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4d/relu_3x3_reduce 1x144x64x128x144"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 144 -k 144 -F 1 -t %T% -V %V%
echo "inception_4d/3x3 10x144x14x14x288x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 144 -k 288 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_4d/relu_3x3 1x288x64x128x288"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 288 -k 288 -F 1 -t %T% -V %V%
echo "inception_4d/5x5_reduce 1x512x64x128x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4d/relu_5x5_reduce 1x32x64x128x32"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 32 -k 32 -F 1 -t %T% -V %V%
echo "inception_4d/5x5 10x32x14x14x64x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 32 -k 64 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_4d/relu_5x5 1x64x64x128x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_4d/pool 10x512x14x14x512x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 14 -H 14 -c 512 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_4d/pool_proj 1x512x64x128x64x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 512 -k 64 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4d/relu_pool_proj 1x64x64x128x64"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 64 -k 64 -F 1 -t %T% -V %V%
echo "inception_4d/output 3"
rem goto end
:4e
echo "inception_4e/1x1 1x528x64x128x256x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 528 -k 256 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4e/relu_1x1 1x256x64x128x256"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 256 -k 256 -F 1 -t %T% -V %V%
echo "inception_4e/3x3_reduce 1x528x64x128x160x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 528 -k 160 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4e/relu_3x3_reduce 1x160x64x128x160"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 160 -k 160 -F 1 -t %T% -V %V%
echo "inception_4e/3x3 10x160x14x14x320x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 160 -k 320 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_4e/relu_3x3 1x320x64x128x320"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 320 -k 320 -F 1 -t %T% -V %V%
echo "inception_4e/5x5_reduce 1x528x64x128x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 528 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4e/relu_5x5_reduce 1x32x64x128x32"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 32 -k 32 -F 1 -t %T% -V %V%
echo "inception_4e/5x5 10x32x14x14x128x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 32 -k 128 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_4e/relu_5x5 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_4e/pool 10x528x14x14x528x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 14 -H 14 -c 528 -k 528 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_4e/pool_proj 1x528x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 528 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_4e/relu_pool_proj 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 14 -H 14 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_4e/output 3"
echo "pool4/3x3_s2 10x832x14x14x832x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 14 -H 14 -c 832 -k 832 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1 -t %T% -V %V%
echo "inception_5a/1x1 1x832x64x128x256x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 256 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5a/relu_1x1 1x256x64x128x256"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 256 -k 256 -F 1 -t %T% -V %V%
echo "inception_5a/3x3_reduce 1x832x64x128x160x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 160 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5a/relu_3x3_reduce 1x160x64x128x160"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 160 -k 160 -F 1 -t %T% -V %V%
echo "inception_5a/3x3 10x160x7x7x320x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 160 -k 320 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_5a/relu_3x3 1x320x64x128x320"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 320 -k 320 -F 1 -t %T% -V %V%
echo "inception_5a/5x5_reduce 1x832x64x128x32x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 32 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5a/relu_5x5_reduce 1x32x64x128x32"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 32 -k 32 -F 1 -t %T% -V %V%
echo "inception_5a/5x5 10x32x7x7x128x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 32 -k 128 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_5a/relu_5x5 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_5a/pool 10x832x7x7x832x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 7 -H 7 -c 832 -k 832 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_5a/pool_proj 1x832x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5a/relu_pool_proj 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 64 -k 128 -F 1 -t %T% -V %V%
echo "inception_5a/output 3"
echo "inception_5b/1x1 1x832x64x128x384x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 384 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5b/relu_1x1 1x384x64x384x256"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 384 -k 384 -F 1 -t %T% -V %V%
echo "inception_5b/3x3_reduce 1x832x64x128x192x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 192 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5b/relu_3x3_reduce 1x192x64x128x192"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 192 -k 192 -F 1 -t %T% -V %V%
echo "inception_5b/3x3 10x192x7x7x384x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 192 -k 384 -x 3 -y 3 -p 1 -q 1  -F 1 -t %T% -V %V% -s %S%
echo "inception_5b/relu_3x3 1x384x64x128x384"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 384 -k 384 -F 1 -t %T% -V %V%
echo "inception_5b/5x5_reduce 1x832x64x128x48x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 48 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5b/relu_5x5_reduce 1x48x64x128x48"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 48 -k 48 -F 1 -t %T% -V %V%
echo "inception_5b/5x5 10x48x7x7x128x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 48 -k 128 -x 5 -y 5 -p 2 -q 2  -F 1 -t %T% -V %V% -s %S%
echo "inception_5b/relu_5x5 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_5b/pool 10x832x7x7x832x3x3"
.\driver\Debug\MIOpenDriver.exe pool -n %B% -W 7 -H 7 -c 832 -k 832 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t %T% -V %V%
echo "inception_5b/pool_proj 1x832x64x128x128x1x1"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 832 -k 128 -x 1 -y 1 -p 0 -q 0  -F 1 -t %T% -V %V% -s %S%
echo "inception_5b/relu_pool_proj 1x128x64x128x128"
rem .\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 7 -H 7 -c 128 -k 128 -F 1 -t %T% -V %V%
echo "inception_5b/output 3"
:end
cd ../src/OCL