from colossalai.amp import AMP_TYPE

#dataset
data_set = 'IMNET'
data_path = 'data/imagenet/ILSVRC2012/raw-data/imagenet-data'
# PVT
model = 'pvt_medium'
drop_path = 0.3
clip_grad = 1.0
batch_size = 100
num_workers = 4
drop = 0.0
lr = 5e-4
opt = 'adamw'
weight_decay = 0.05
momentum = 0.9
num_epochs = 2
nb_classes = 1000

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

gradient_accumulation = 16
use_mcloader = False
inat_category = 'name'

input_size = 224
color_jitter = 0.4
aa = 'rand-m9-mstd0.5-inc1'
train_interpolation = 'bicubic'
reprob = 0.25
remode = 'pixel'
recount = 1
pin_mem = True
mixup = 0.8
cutmix = 1.0
cutmix_minmax = None
mixup_prob = 1.0
mixup_switch_prob = 0.5
mixup_mode = 'batch'
smoothing = 0.1
clip_grad = None
output_dir = ''
finetune = ''
fp32_resume = False

