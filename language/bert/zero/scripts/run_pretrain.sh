
BERT_CONFIG_PATH='./configs/bert_base.json'
PY_FILE_PATH='./pretraining/run_pretraining.py'
DATA_PATH='./pretrain/phase1/unbinned/parquet'
VOCAB_FILE='bert-base-uncased'

export PYTHONPATH=$PWD

colossalai run --nproc_per_node 1 \
    --master_port 29550 \
    $PY_FILE_PATH \
    --bert-config $BERT_CONFIG_PATH \
    --lr 1e-4 \
    --data $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --batch-size 32 \
    --epoch 100
