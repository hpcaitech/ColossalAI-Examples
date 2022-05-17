#!/bin/bash
export LOCAL_RANK=0
export NUM_NODES=1
export NPROC_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=19198

sh scripts/launch_BigGAN.sh