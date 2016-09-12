cd ../bin
goto fullconvnet
echo "CIFAR-10"
echo "conv1 100x3x32x32x32x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 100 -iw 32 -ih 32 -ic0 3 -oc0 32 -fs 5 -cnv -fw -sch
echo "conv1 100x32x16x16x32x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 100 -iw 16 -ih 16 -ic0 32 -oc0 32 -fs 5 -cnv -fw -sch
echo "conv1 100x32x8x8x64x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 100 -iw 8 -ih 8 -ic0 32 -oc0 64 -fs 5 -cnv -fw -sch
echo "conv1 200x3x32x32x32x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 200 -iw 32 -ih 32 -ic0 3 -oc0 32 -fs 5 -cnv -fw -sch
echo "conv1 200x32x16x16x32x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 200 -iw 16 -ih 16 -ic0 32 -oc0 32 -fs 5 -cnv -fw -sch
echo "conv1 200x32x8x8x64x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 200 -iw 8 -ih 8 -ic0 32 -oc0 64 -fs 5 -cnv -fw -sch
echo "conv1 1x3x32x32x32x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 1 -iw 32 -ih 32 -ic0 3 -oc0 32 -fs 5 -cnv -fw -sch
echo "conv1 1x32x16x16x32x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 1 -iw 16 -ih 16 -ic0 32 -oc0 32 -fs 5 -cnv -fw -sch
echo "conv1 1x32x8x8x64x5x5"
aLibDNNDriver.exe -li 500 -ni 3 -n -l -bz 1 -iw 8 -ih 8 -ic0 32 -oc0 64 -fs 5 -cnv -fw -sch
:end
cd ../aLibDNN