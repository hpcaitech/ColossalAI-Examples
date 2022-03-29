BATCH_SIZE = 16
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-2

TENSOR_PARALLEL_SIZE = 2
TENSOR_PARALLEL_MODE = '1d'

NUM_EPOCHS = 800
WARMUP_EPOCHS = 40
clip_max_norm = 2.

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

seed = 77

LOG_PATH = f"./detr_{TENSOR_PARALLEL_MODE}_ai2d_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}/"


find_unused_parameters = True

coco_path = '/data/huxin/xjtuhx/projects/ai2d-detection-baselines/111/data_dir/ai2d/'
save_ckpt_freq = 50
lr_backbone = 1e-5
device = 'cuda'
lr_drop = 200
backbone = 'resnet34'
dilation = None
position_embedding = 'sine'
enc_layers = 2
dec_layers = 2
dim_feedforward = 512
hidden_dim = 256
dropout = 0.1
nheads = 1
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
dataset_file = 'ai2d'
remove_difficult = True
output_dir = '/data/huxin/xjtuhx/projects/ai2d-detection-baselines/111/output_test/'
resume = ''
start_epoch = 0
eval = False
num_workers = 2
world_size = 1
dist_url = 'env://'
distributed = True
aux_loss = False
















