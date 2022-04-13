#export DATA=$PWD/data
torchrun --standalone --nnodes=1 --nproc_per_node 1 run_resnet_property_prediction_with_engine.py
#torchrun --standalone --nnodes=1 --nproc_per_node 1  run_resnet_property_prediction_with_trainer.py