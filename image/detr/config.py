BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '1d'

NUM_EPOCHS = 300
lr_drop = 200
clip_max_norm = 0.1

# gradient_clipping = 0.1

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

cudnn_benchmark = False

seed = 42

LOG_PATH = f"./detr_{TENSOR_PARALLEL_MODE}_coco_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}/"


# find_unused_parameters = True

coco_path = '/data/scratch/coco'
save_ckpt_freq = 50
lr_backbone = 1e-5
device = 'cuda'
lr_drop = 200
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.1
nheads = 8
num_queries = 100
masks = False
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2
mask_loss_coef = 1
dice_loss_coef = 1
bbox_loss_coef = 5
giou_loss_coef = 2
eos_coef = 0.1
dataset_file = 'coco'
remove_difficult = False
output_dir = ''
resume = ''
start_epoch = 0
eval = False
num_workers = 2
dist_url = 'env://'
distributed = True
aux_loss = True
















