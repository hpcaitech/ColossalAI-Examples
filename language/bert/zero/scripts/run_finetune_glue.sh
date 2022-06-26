
GLUE_DATASET=$PWD/download/glue/MRPC
VOCAB_FILE="bert-base-uncased"
CODE_DIR=$PWD/finetuning/glue

colossalai run --nproc_per_node 4 \
    --master_port 29510 \
    $CODE_DIR/main.py \
    --data_dir $GLUE_DATASET \
    --task_name mrpc \
    --bert_config ./configs/bert_base.json \
    --vocab_file $VOCAB_FILE \
    --output_dir ./out \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --num_train_epochs 3 \
    --train \
    --eval \
    --predict 
