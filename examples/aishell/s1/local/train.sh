#! /usr/bin/env bash

if [ $# != 1 ];then
    echo "usage: ${0} ckpt_tag"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

python3 -u ${BIN_DIR}/train.py \
--device 'gpu' \
--nproc ${ngpu} \
--config conf/conformer.yaml \
--output ckpt-${1}


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0
