#!/bin/bash
DATAROOT="./data"
# [ -e $DATAROOT ] || mkdir $DATAROOT

urls=("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")

git lfs install
git clone https://huggingface.co/datasets/Minter/MNIST data

for url in ${urls[@]}; do
  gz=$(basename $url)
  ugz=$(echo $gz | sed 's/\.gz//')
  # [ -e $DATAROOT/$gz ] || wget -c -O $DATAROOT/$gz $url || curl -LO $DATAROOT/$gz $url
  [ -e $DATAROOT/$ugz ] || gunzip -k $DATAROOT/$gz
done