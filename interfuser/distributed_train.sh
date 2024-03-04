#!/bin/bash
NUM_PROC=$1
shift
echo training
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC  --master_port 40300 train.py "$@"