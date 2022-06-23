#!/bin/bash

WORK_DIR=$(pwd)
DATA_ROOT=$1/arrow
NUM_GPUS=$2

if [ -z $DATA_ROOT ] || [ -z $NUM_GPUS ]
then
    echo "Usage: $0 <data_root> <num_gpus>"
    exit 1
fi

cd $WORK_DIR

if ! [ -x "$(command -v mpirun)" ]
then
    torchrun --nproc_per_node $NUM_GPUS --master_addr localhost --master_port 11455 run.py
else
    mpirun -np $NUM_GPUS python run.py with data_root=$DATA_ROOT num_gpus=$NUM_GPUS num_nodes=1 task_mlm_itm_s step200k per_gpu_batchsize=96
fi
