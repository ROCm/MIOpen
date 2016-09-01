cd ../bin
echo "GoogleNet"
echo "conv1/7x7_s2 10x3x224x224x64x7x7"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 224 -ih 224 -ic0 3 -oc0 64 -fs 7 -cnv -fw -sd 2
echo "conv1/relu_7x7 1x64x112x112x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 1024 -ih 512 -ic0 64 -oc0 64 -nrn 3 -fw
echo "pool1/3x3_s2 10x64x112x112x64x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 112 -ih 112 -ic0 64 -oc0 64 -fs 3 -plg -sd 2 -fw
echo "pool1/norm 1x64x56x56x64x5x5"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 512 -ih 256 -ic0 64 -oc0 192 -fs 5 -lrn 0 -fw
echo "conv2/3x3_reduce 1x64x56x56x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 512 -ih 256 -ic0 64 -oc0 64 -fs 1 -nopad -cnv -fw -sch
echo "conv2/relu_3x3_reduce 1x64x256x512x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 512 -ih 256 -ic0 64 -oc0 64 -nrn 3 -fw
echo "conv2/3x3 10x64x56x56x192x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 64 -ih 56 -ic0 56 -oc0 192 -fs 3 -cnv -fw -sch
echo "conv2/relu_3x3 1x192x256x512x192"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 512 -ih 256 -ic0 192 -oc0 192 -nrn 3 -fw
echo "conv2/norm2 1x192x256x512x192x5x5"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 56 -ih 56 -ic0 192 -oc0 192 -fs 5 -lrn 0 -fw
echo "pool2/3x3_s2 10x192x56x56x192x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 56 -ih 56 -ic0 192 -oc0 192 -fs 3 -plg -sd 2 -fw
echo "inception_3a/1x1 1x192x28x28x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 192 -oc0 64 -fs 1 -nopad -cnv -fw -sch
echo "inception_3a/3x3_reduce 1x192x128x256x96x3x3"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 192 -oc0 96 -fs 1 -nopad -cnv -fw -sch
echo "inception_3a/relu_3x3_reduce 1x96x128x256x96"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 96 -oc0 96 -nrn 3 -fw
echo "inception_3a/3x3 10x96x28x28x128x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 28 -ih 28 -ic0 96 -oc0 128 -fs 3 -cnv -fw -sch
echo "inception_3a/relu_3x3 1x128x128x256x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_3a/5x5_reduce 1x192x128x256x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 192 -oc0 16 -fs 1 -nopad -cnv -fw -sch
echo "inception_3a/relu_5x5_reduce 1x16x128x256x16"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 16 -oc0 16 -nrn 3 -fw
echo "inception_3a/5x5 10x16x28x28x32x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 28 -ih 28 -ic0 16 -oc0 32 -fs 5 -cnv -fw -sch
echo "inception_3a/relu_5x5 1x32x128x256x32"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 32 -oc0 32 -nrn 3 -fw
echo "inception_3a/pool 10x192x28x28x192x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 28 -ih 28 -ic0 192 -oc0 192 -fs 3 -plg -sd 1 -fw
echo "inception_3a/pool_proj 1x192x128x256x32x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 192 -oc0 32 -fs 1 -nopad -cnv -fw -sch
echo "inception_3a/relu_pool_proj 1x32x128x256x32"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 32 -oc0 32 -nrn 3 -fw
echo "inception_3a/output 1x128x256x256"
rem goto end
:3b
echo "inception_3b/1x1 1x256x128x256x128x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 256 -oc0 128 -fs 1 -nopad -cnv -fw -sch
echo "nception_3b/relu_1x1 1x128x128x256x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_3b/3x3_reduce 1x256x128x256x128x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 256 -oc0 128 -fs 1 -nopad -cnv -fw -sch
echo "inception_3a/relu_3x3_reduce 1x128x128x256x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_3b/3x3 10x128x28x28x192x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 28 -ih 28 -ic0 128 -oc0 192 -fs 3 -cnv -fw -sch
echo "inception_3b/relu_3x3 1x192x128x256x192"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 192 -oc0 192 -nrn 3 -fw
echo "inception_3b/5x5_reduce 1x256x128x256x32x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 256 -oc0 32 -fs 1 -nopad -cnv -fw -sch
echo "inception_3b/relu_5x5_reduce 1x32x128x256x32"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 32 -oc0 32 -nrn 3 -fw
echo "inception_3b/5x5 10x32x28x28x96x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 28 -ih 28 -ic0 32 -oc0 96 -fs 5 -cnv -fw -sch
echo "inception_3b/relu_5x5 1x96x128x256x96"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 96 -oc0 96 -nrn 3 -fw
echo "inception_3b/pool 10x256x28x28x256x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 28 -ih 28 -ic0 256 -oc0 256 -fs 3 -plg -sd 1 -fw
echo "inception_3b/pool_proj 1x256x128x256x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 256 -oc0 64 -fs 1 -nopad -cnv -fw -sch
echo "inception_3b/relu_pool_proj 1x64x128x256x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 256 -ih 128 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_3b/output 3"
echo "pool3/3x3_s2 1x480x28x28x480x3x3"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 28 -ih 28 -ic0 480 -oc0 480 -fs 3 -plg -sd 2 -fw
rem goto end
:4a
echo "inception_4a/1x1 1x480x64x128x192x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 480 -oc0 192 -fs 1 -nopad -cnv -fw -sch
echo "nception_4a/relu_1x1 1x192x64x128x192x3x3"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 192 -oc0 192 -nrn 3 -fw
echo "inception_4a/3x3_reduce 1x480x64x128x96x3x3"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 480 -oc0 96 -fs 1 -nopad -cnv -fw -sch
echo "inception_4a/relu_3x3_reduce 1x96x64x128x96"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 96 -oc0 96 -nrn 3 -fw
echo "inception_4a/3x3 10x96x14x14x208x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 96 -oc0 208 -fs 3 -cnv -fw -sch
echo "inception_4a/relu_3x3 1x208x64x128x208"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 208 -oc0 208 -nrn 3 -fw
echo "inception_4a/5x5_reduce 1x480x64x128x16x1s1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 480 -oc0 16 -fs 1 -nopad -cnv -fw -sch
echo "inception_4a/relu_5x5_reduce 1x16x64x128x16"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 16 -oc0 16 -nrn 3 -fw
echo "inception_4a/5x5 10x16x14x14x48x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 16 -oc0 48 -fs 5 -cnv -fw -sch
echo "inception_4a/relu_5x5_reduce 1x48x64x128x48"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 48 -oc0 48 -nrn 3 -fw
echo "inception_4a/pool 10x480x14x14x480x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 480 -oc0 480 -fs 3 -plg -sd 1 -fw
echo "inception_4a/pool_proj 1x480x64x128x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 480 -oc0 64 -fs 1 -nopad -cnv -fw -sch
echo "inception_4a/relu_pool_proj 1x64x64x128x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_4a/output 3"
rem goto end
:4b
echo "inception_4b/1x1 1x512x64x128x160x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 160 -fs 1 -nopad -cnv -fw -sch
echo "inception_4b/relu_1x1 1x160x64x128x160"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 160 -oc0 160 -nrn 3 -fw
echo "inception_4b/3x3_reduce 1x512x64x128x112x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 112 -fs 1 -nopad -cnv -fw -sch
echo "inception_4b/relu_3x3_reduce 1x112x64x128x112"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 112 -oc0 112 -nrn 3 -fw
echo "inception_4b/3x3 10x112x14x14x224x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 112 -oc0 224 -fs 3 -cnv -fw -sch
echo "inception_4b/relu_3x3 1x224x64x128x224"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 224 -oc0 224 -nrn 3 -fw
echo "inception_4b/5x5_reduce 1x512x64x128x24x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 128 -ic0 512 -oc0 24 -fs 1 -nopad -cnv -fw -sch
echo "inception_4b/relu_5x5_reduce 1x24x64x128x24"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 24 -oc0 24 -nrn 3 -fw
echo "inception_4b/5x5 10x24x14x14x64x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 24 -oc0 64 -fs 5 -cnv -fw -sch
echo "inception_4b/relu_5x5 1x64x64x128x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_4b/pool 10x512x14x14x512x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 512 -oc0 512 -fs 3 -plg -sd 1 -fw
echo "inception_4b/pool_proj 1x512x64x128x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 64 -fs 1 -nopad -cnv -fw -sch
echo "inception_4b/relu_pool_proj 1x64x64x128x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_4b/output 3"
echo "inception_4c/1x1 1x512x64x128x128x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 128 -fs 1 -nopad -cnv -fw -sch
echo "inception_4c/relu_1x1 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_4c/3x3_reduce 1x512x64x128x128x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 128 -fs 1 -nopad -cnv -fw -sch
echo "inception_4c/relu_3x3_reduce 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_4c/3x3 10x128x14x14x256x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 128 -oc0 256 -fs 3 -cnv -fw -sch
echo "inception_4c/relu_3x3 1x256x64x128x256"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 256 -oc0 256 -nrn 3 -fw
echo "inception_4c/5x5_reduce 1x512x64x128x24x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 24 -fs 1 -nopad -cnv -fw -sch
echo "inception_4c/relu_5x5_reduce 1x24x64x128x24"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 24 -oc0 24 -nrn 3 -fw
echo "inception_4c/5x5 10x24x14x14x64x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 24 -oc0 64 -fs 5 -cnv -fw -sch
echo "inception_4c/relu_5x5 1x64x64x128x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_4c/pool 10x512x14x14x512x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 512 -oc0 512 -fs 3 -plg -sd 1 -fw
echo "inception_4c/pool_proj 1x512x64x128x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 64 -fs 1 -nopad -cnv -fw -sch
echo "inception_4c/relu_pool_proj 1x64x64x128x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_4c/output 3"
echo "inception_4d/1x1 1x512x64x128x112x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 112 -fs 1 -nopad -cnv -fw -sch
echo "inception_4d/relu_1x1 1x112x64x128x112"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 112 -oc0 112 -nrn 3 -fw
echo "inception_4d/3x3_reduce 1x512x64x128x144x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 144 -fs 1 -nopad -cnv -fw -sch
echo "inception_4d/relu_3x3_reduce 1x144x64x128x144"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 144 -oc0 144 -nrn 3 -fw
echo "inception_4d/3x3 10x144x14x14x288x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 144 -oc0 288 -fs 3 -cnv -fw -sch
echo "inception_4d/relu_3x3 1x288x64x128x288"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 288 -oc0 288 -nrn 3 -fw
echo "inception_4d/5x5_reduce 1x512x64x128x32x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 32 -fs 1 -nopad -cnv -fw -sch
echo "inception_4d/relu_5x5_reduce 1x32x64x128x32"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 32 -oc0 32 -nrn 3 -fw
echo "inception_4d/5x5 10x32x14x14x64x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 32 -oc0 64 -fs 5 -cnv -fw -sch
echo "inception_4d/relu_5x5 1x64x64x128x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_4d/pool 10x512x14x14x512x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 512 -oc0 512 -fs 3 -plg -sd 1 -fw
echo "inception_4d/pool_proj 1x512x64x128x64x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 512 -oc0 64 -fs 1 -nopad -cnv -fw -sch
echo "inception_4d/relu_pool_proj 1x64x64x128x64"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 64 -oc0 64 -nrn 3 -fw
echo "inception_4d/output 3"
rem goto end
:4e
echo "inception_4e/1x1 1x528x64x128x256x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 528 -oc0 256 -fs 1 -nopad -cnv -fw -sch
echo "inception_4e/relu_1x1 1x256x64x128x256"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 256 -oc0 256 -nrn 3 -fw
echo "inception_4e/3x3_reduce 1x528x64x128x160x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 528 -oc0 160 -fs 1 -nopad -cnv -fw -sch
echo "inception_4e/relu_3x3_reduce 1x160x64x128x160"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 160 -oc0 160 -nrn 3 -fw
echo "inception_4e/3x3 10x160x14x14x320x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 160 -oc0 320 -fs 3 -cnv -fw -sch
echo "inception_4e/relu_3x3 1x320x64x128x320"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 320 -oc0 320 -nrn 3 -fw
echo "inception_4e/5x5_reduce 1x528x64x128x32x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 528 -oc0 32 -fs 1 -nopad -cnv -fw -sch
echo "inception_4e/relu_5x5_reduce 1x32x64x128x32"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 32 -oc0 32 -nrn 3 -fw
echo "inception_4e/5x5 10x32x14x14x128x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 32 -oc0 128 -fs 5 -cnv -fw -sch
echo "inception_4e/relu_5x5 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_4e/pool 10x528x14x14x528x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 528 -oc0 528 -fs 3 -plg -sd 1 -fw
echo "inception_4e/pool_proj 1x528x64x128x128x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 528 -oc0 128 -fs 1 -nopad -cnv -fw -sch
echo "inception_4e/relu_pool_proj 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_4e/output 3"
echo "pool4/3x3_s2 10x832x14x14x832x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 14 -ih 14 -ic0 832 -oc0 832 -fs 3 -plg -sd 2 -fw
echo "inception_5a/1x1 1x832x64x128x256x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 256 -fs 1 -nopad -cnv -fw -sch
echo "inception_5a/relu_1x1 1x256x64x128x256"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 256 -oc0 256 -nrn 3 -fw
echo "inception_5a/3x3_reduce 1x832x64x128x160x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 160 -fs 1 -nopad -cnv -fw -sch
echo "inception_5a/relu_3x3_reduce 1x160x64x128x160"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 160 -oc0 160 -nrn 3 -fw
echo "inception_5a/3x3 10x160x7x7x320x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 7 -ih 7 -ic0 160 -oc0 320 -fs 3 -cnv -fw -sch
echo "inception_5a/relu_3x3 1x320x64x128x320"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 320 -oc0 320 -nrn 3 -fw
echo "inception_5a/5x5_reduce 1x832x64x128x32x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 32 -fs 1 -nopad -cnv -fw -sch
echo "inception_5a/relu_5x5_reduce 1x32x64x128x32"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 32 -oc0 32 -nrn 3 -fw
echo "inception_5a/5x5 10x32x7x7x128x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 7 -ih 7 -ic0 32 -oc0 128 -fs 5 -cnv -fw -sch
echo "inception_5a/relu_5x5 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_5a/pool 10x832x7x7x832x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 7 -ih 7 -ic0 832 -oc0 832 -fs 3 -plg -sd 1 -fw
echo "inception_5a/pool_proj 1x832x64x128x128x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 128 -fs 1 -nopad -cnv -fw -sch
echo "inception_5a/relu_pool_proj 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_5a/output 3"
echo "inception_5b/1x1 1x832x64x128x384x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 384 -fs 1 -nopad -cnv -fw -sch
echo "inception_5b/relu_1x1 1x384x64x384x256"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 384 -oc0 384 -nrn 3 -fw
echo "inception_5b/3x3_reduce 1x832x64x128x192x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 192 -fs 1 -nopad -cnv -fw -sch
echo "inception_5b/relu_3x3_reduce 1x192x64x128x192"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 192 -oc0 192 -nrn 3 -fw
echo "inception_5b/3x3 10x192x7x7x384x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 7 -ih 7 -ic0 192 -oc0 384 -fs 3 -cnv -fw -sch
echo "inception_5b/relu_3x3 1x384x64x128x384"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 384 -oc0 384 -nrn 3 -fw
echo "inception_5b/5x5_reduce 1x832x64x128x48x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 48 -fs 1 -nopad -cnv -fw -sch
echo "inception_5b/relu_5x5_reduce 1x48x64x128x48"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 48 -oc0 48 -nrn 3 -fw
echo "inception_5b/5x5 10x48x7x7x128x5x5"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 7 -ih 7 -ic0 48 -oc0 128 -fs 5 -cnv -fw -sch
echo "inception_5b/relu_5x5 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_5b/pool 10x832x7x7x832x3x3"
aLibDNNDriver -li 500 -ni 3 -n -l -bz 10 -iw 7 -ih 7 -ic0 832 -oc0 832 -fs 3 -plg -sd 1 -fw
echo "inception_5b/pool_proj 1x832x64x128x128x1x1"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 832 -oc0 128 -fs 1 -nopad -cnv -fw -sch
echo "inception_5b/relu_pool_proj 1x128x64x128x128"
rem aLibDNNDriver -li 500 -ni 3 -n -l -bz 1 -iw 128 -ih 64 -ic0 128 -oc0 128 -nrn 3 -fw
echo "inception_5b/output 3"
:end
cd ../aLibDNN