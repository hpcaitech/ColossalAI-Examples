export BS=${1:-16}
export MEMCAP=${2:-0}
export MODEL=${3:-"6.7b"}
export GPUNUM=${4:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# make directory for logs
mkdir -p ./logs

# env PYTORCH_NO_CUDA_MEMORY_CACHING=1 
torchrun \
  --nproc_per_node ${GPUNUM} \
  --master_port 19198 \
  run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --model_name_or_path facebook/opt-${MODEL} \
  --output_dir $PWD \
  --mem_cap ${MEMCAP} \
  --per_device_train_batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log

