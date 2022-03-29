export DATA=/data/scratch/gpt_data/small-gpt-dataset.json
export CUDA_LAUNCH_BLOCKING=1
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_gpt.py --from_torch --config gpt2_configs/gpt2_zero3.py