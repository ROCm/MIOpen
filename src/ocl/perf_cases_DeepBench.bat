set S=%1
set V=%2
set T=%3
cd ../../build
PATH=.\src\Debug;%PATH%
echo "DeepBench"
echo "conv1_2 32x1x161x700x32x5x20"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n 32 -H 161 -W 700 -c 1 -k 32 -x 20 -y 5 -p 0 -q 0 -u 2 -v 2 -F 1 -t %T% -s %S% -V %V%
eecho "conv2 8x64x54x54x64x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n 8 -H 54 -W 54 -c 64 -k 64 -x 3 -y 3 -p 1 -q 1 -F 0 -t %T% -s %S% -V %V%
echo "conv3 16x3x224x224x64x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n 16 -H 224 -W 224 -c 3 -k 64 -x 3 -y 3 -p 1 -q 1 -F 0 -t %T% -s %S% -V %V%
echo "conv4 16x512x7x7x512x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n 16 -H 7 -W 7 -c 512 -k 512 -x 3 -y 3 -p 1 -q 1 -F 0 -t %T% -s %S% -V %V%
echo "conv5 16x192x28x28x32x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n 16 -H 28 -W 28 -c 192 -k 32 -x 5 -y 5 -p 2 -q 2 -F 0 -t %T% -s %S% -V %V%
:end
cd ../src/OCL