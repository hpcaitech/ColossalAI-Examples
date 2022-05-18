 #!/usr/bin/env bash 

phase=$1
vocab_file=$2
seq_length=$3
num_dask_workers=${4:-32}
wikipedia_source=${5:-$PWD/wikipedia/source}
seed=${6:-12439}
num_shards_per_worker=${7:-32}
num_workers=${8:-4}
num_gpus=${9:-$(nvidia-smi -L | wc -l)}
sample_ratio=${10:-0.9}
masking=${11:-static}
phase2_bin_size=${12:-64}

readonly num_blocks=$((num_shards_per_worker * $(( num_workers > 0 ? num_workers : 1 )) * num_gpus))

if [ $num_blocks == 0 ]; then
   echo "Error! Number of blocks is 0, but expected to be larger than 0."
   exit -1
fi

# ensure phase is correct
if [ "${phase}" == "1" ]; then
   DATASET=pretrain/phase1/unbinned/parquet
   OUTPUT_DATA_DIR=${21:-$PWD/${DATASET}/}
elif [ "${phase}" == "2" ]; then
   DATASET=pretrain/phase2/bin_size_${phase2_bin_size}/parquet
   OUTPUT_DATA_DIR=${22:-$PWD/${DATASET}/}
else
   echo "Error! Got invalid phase ${phase}, expected 1 or 2"
   exit -1
fi

# check for jemalloc path
# install by conda install jemalloc
jemalloc_path=$(which python | sed 's/bin\/python//' | awk '{print $1"lib/libjemalloc.so"}')

if [ ! -f "${jemalloc_path}" ]; then
   echo "jemalloc does not exist in ${jemalloc_path}, please install via conda install jemalloc"
   exit -1
fi

# check for wikipedia data
if [ "${wikipedia_source}" == "" ]; then
   echo "Error! wikipedia source is not given"
   exit -1
elif [ ! -d "${wikipedia_source}" ]; then
   echo "Error, wikipedia source does not exist in ${wikipedia_source}"
   exit -1
fi

# check for masking type
if [ "${masking}" == "static" ]; then
   readonly masking_flag="--masking"
elif [ "${masking}" == "dynamic" ]; then
   readonly masking_flag=""
else
   echo "Error! masking=${masking} not supported!"
   exit -1
fi

# check for phase 2 bin size
if [ "${phase}" == "1" ]; then
   readonly phase2_bin_size_flag=""
elif [ "${phase2_bin_size}" == "none" ]; then
   readonly phase2_bin_size_flag=""
elif [[ "${phase2_bin_size}" =~ ^(32|64|128|256|512)$ ]]; then
   readonly phase2_bin_size_flag="--bin-size ${phase2_bin_size}"
else
   echo "Error! phase2_bin_size=${phase2_bin_size} not supported!"
   exit -1
fi

# check if the parquets files already exists
count=`ls -1 ${OUTPUT_DATA_DIR}/*.parquet 2>/dev/null | wc -l`

if [ $count != 0 ]; then
   echo "Error! Parquet files already exist in ${OUTPUT_DATA_DIR}"
   exit -1
fi

mpirun --oversubscribe \
    --allow-run-as-root \
    -np ${num_dask_workers} \
    -x LD_PRELOAD=${jemalloc_path} \
    preprocess_bert_pretrain \
      --schedule mpi \
      --vocab-file ${vocab_file} \
      --wikipedia ${wikipedia_source} \
      --sink ${OUTPUT_DATA_DIR} \
      --target-seq-length ${seq_length} \
      --num-blocks ${num_blocks} \
      --sample-ratio ${sample_ratio} \
      ${phase2_bin_size_flag} \
      ${masking_flag} \
      --seed ${seed}

mpirun --oversubscribe \
   --allow-run-as-root \
   -np ${num_dask_workers} \
   balance_dask_output \
      --indir ${OUTPUT_DATA_DIR} \
      --num-shards ${num_blocks}