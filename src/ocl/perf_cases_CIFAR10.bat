rem arguments:
rem D - MiOpenDriver location
set D=%1
echo "CIFAR-10"
echo "conv1 100x3x32x32x32x5x5"
%D%\MIOpenDriver.exe conv -i 1 -n 100 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -F 0 -V 1 -t 1 -s 0 
echo "pool1 100x32x32x32x32x3x3"
%D%\MIOpenDriver.exe pool -i 1 -n 100 -W  32 -H 32 -c 32 -k 32 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 0 -V 1 -t 1
echo "conv2 100x32x16x16x32x5x5"
%D%\MIOpenDriver.exe conv -i 1 -n 100 -W 16 -H 16 -c 32 -k 32 -x 5 -y 5 -p 2 -q 2 -F 0 -V 1 -t 1 -s 0
echo "pool2 100x32x16x16x3x3"
%D%\MIOpenDriver.exe pool -i 1 -n 100  -W  32 -H 32 -c 32 -k 32 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 0 -V 1 -t 1
echo "conv3 100x32x8x8x64x5x5"
%D%\MIOpenDriver.exe conv -i 1 -n 100 -W 8 -H 8 -c 32 -k 64 -x 5 -y 5 -p 2 -q 2 -F 0 -V 1 -t 1 -s 0
