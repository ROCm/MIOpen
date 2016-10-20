set B=%1
set S=%2
set V=%3
set T=%4
cd ../../build
PATH=.\src\Debug;%PATH%
echo "CIFAR-10"
echo "conv1 100x3x32x32x32x5x5"
.\driver\Debug\MLOpenDriver.exe conv -i 1 -n %B% -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "pool1 100x32x32x32x32x3x3"
.\driver\Debug\MLOpenDriver.exe pool -t %T% -i 1 -n %B%  -V %V% -W  32 -H 32 -c 32 -k 32 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1
echo "conv2 100x32x16x16x32x5x5"
.\driver\Debug\MLOpenDriver.exe conv -i 1 -n %B% -W 16 -H 16 -c 32 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "pool2 100x32x16x16x3x3"
.\driver\Debug\MLOpenDriver.exe pool -t %T% -i 1 -n %B%  -V %V% -W  32 -H 32 -c 32 -k 32 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2 -F 1
echo "conv3 100x32x8x8x64x5x5"
.\driver\Debug\MLOpenDriver.exe conv -i 1 -n %B% -W 8 -H 8 -c 32 -k 64 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
:end
cd ../src/OCL