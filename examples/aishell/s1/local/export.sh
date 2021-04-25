#! /usr/bin/env bash

if [ $# != 2 ];then
    echo "usage: ${0}  ckpt_path jit_model_path"
    exit -1
fi

python3 -u ${BIN_DIR}/export.py \
--config conf/conformer.yaml \
--checkpoint_path ${1} \
--export_path ${2} 


if [ $? -ne 0 ]; then
    echo "Failed in export!"
    exit 1
fi


exit 0
