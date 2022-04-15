from colossalai.amp import AMP_TYPE

BATCH_SIZE = 32
NUM_EPOCHS = 20

fp16 = dict(
    mode=AMP_TYPE.TORCH
)

gradient_accumulation = 2

#parallel = dict(
#    data=1,
#    pipeline=1,
#    tensor=dict(size=4, mode='2d')
#)

Neighbor_Sample_Size = 8
N_Iter = 1
Ir = 0.005