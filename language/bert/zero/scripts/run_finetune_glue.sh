
GLUE_DATASET=/home/lclsg/projects/colossal/github/lsg/ColossalAI-Examples/language/bert/zero/download/glue/MRPC
VOCAB_FILE="bert-base-uncased"
CODE_DIR=$PWD/finetuning/glue

colossalai run --nproc_per_node 4 \
    --master_port 29505 \
    $CODE_DIR/main.py \
    --data_dir $GLUE_DATASET \
    --task_name mrpc \
    --bert_config ./configs/bert_base.json \
    --vocab_file $VOCAB_FILE \
    --init_checkpoint /home/lclsg/projects/DeepLearningExamples/PyTorch/LanguageModeling/BERT/download/nvidia_pretrained_weights/bert_base.pt \
    --output_dir ./out \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --num_train_epochs 3 \
    --train \
    --eval \
    --predict 

    # --init_checkpoint ./ckpt/bert-base-uncased.bin \
    # --train \
    # --bert_config ./configs/bert_base.json \





    