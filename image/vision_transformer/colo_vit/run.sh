export DATA=/data/scratch/imagenet/tf_records
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=4,5,6,7 colossalai run --nproc_per_node 4 train.py --config configs/vit_1d_tp2.py --master_port 29598 | tee ./out 2>&1
