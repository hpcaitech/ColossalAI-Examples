import os

import torch
from torch.distributed import get_rank, get_world_size
from transformers import ViTConfig, ViTForImageClassification

from zero.common.utils import CONFIG, ModelFromHF

_vit_b = dict(
    img_size=224,
    patch_size=16,
    hidden_size=768,
    intermediate_size=3072,
    num_heads=12,
    depth=12,
    dropout=0.1,
    num_labels=1000,
    numel=86567656,
    checkpoint=False,
    evaluation='acc',
)

_vit_h = dict(
    img_size=224,
    patch_size=16,
    hidden_size=1280,
    intermediate_size=5120,
    num_heads=16,
    depth=32,
    dropout=0.1,
    num_labels=1000,
    numel=632199400,
    checkpoint=True,
    evaluation='acc',
)

_vit_g = dict(
    img_size=224,
    patch_size=14,
    hidden_size=1664,
    intermediate_size=8192,
    num_heads=16,
    depth=48,
    dropout=0.1,
    num_labels=1000,
    numel=1844440680,
    checkpoint=True,
    evaluation='acc',
)

_vit_10b = dict(
    img_size=224,
    patch_size=16,
    hidden_size=4096,
    intermediate_size=16384,
    num_heads=16,
    depth=50,
    dropout=0.1,
    num_labels=1000,
    numel=10077058024,
    checkpoint=True,
    evaluation='acc',
)

_vit_configurations = dict(
    vit=_vit_b,
    vit_b=_vit_b,
    vit_h=_vit_h,
    vit_g=_vit_g,
    vit_10b=_vit_10b,
)

_default_hyperparameters = dict(
    batch_size=4,
    mixup_alpha=0.2,
    learning_rate=3e-3,
    weight_decay=0.3,
    num_epochs=2,
    warmup_epochs=1,
    steps_per_epoch=100,
)


def build_data():
    import glob

    import numpy as np
    import nvidia.dali.fn as fn
    import nvidia.dali.tfrecord as tfrec
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

    class DaliDataloader(DALIClassificationIterator):

        def __init__(self,
                     tfrec_filenames,
                     tfrec_idx_filenames,
                     shard_id=0,
                     num_shards=1,
                     batch_size=128,
                     num_threads=4,
                     resize=256,
                     crop=224,
                     prefetch=2,
                     training=True,
                     gpu_aug=False,
                     cuda=True,
                     mixup_alpha=0.0):
            self.mixup_alpha = mixup_alpha
            self.training = training
            pipe = Pipeline(batch_size=batch_size,
                            num_threads=num_threads,
                            device_id=torch.cuda.current_device() if cuda else None,
                            seed=1024)
            with pipe:
                inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                             index_path=tfrec_idx_filenames,
                                             random_shuffle=training,
                                             shard_id=shard_id,
                                             num_shards=num_shards,
                                             initial_fill=10000,
                                             read_ahead=True,
                                             prefetch_queue_depth=prefetch,
                                             name='Reader',
                                             features={
                                                 'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                                                 'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                             })
                images = inputs["image/encoded"]

                if training:
                    images = fn.decoders.image(images, device='mixed' if gpu_aug else 'cpu', output_type=types.RGB)
                    images = fn.random_resized_crop(images, size=crop, device='gpu' if gpu_aug else 'cpu')
                    flip_lr = fn.random.coin_flip(probability=0.5)
                else:
                    # decode jpeg and resize
                    images = fn.decoders.image(images, device='mixed' if gpu_aug else 'cpu', output_type=types.RGB)
                    images = fn.resize(images,
                                       device='gpu' if gpu_aug else 'cpu',
                                       resize_x=resize,
                                       resize_y=resize,
                                       dtype=types.FLOAT,
                                       interp_type=types.INTERP_TRIANGULAR)
                    flip_lr = False

                # center crop and normalize
                images = fn.crop_mirror_normalize(images,
                                                  dtype=types.FLOAT,
                                                  crop=(crop, crop),
                                                  mean=[127.5],
                                                  std=[127.5],
                                                  mirror=flip_lr)
                label = inputs["image/class/label"] - 1    # 0-999
                # LSG: element_extract will raise exception, let's flatten outside
                # label = fn.element_extract(label, element_map=0)  # Flatten
                if cuda:    # transfer data to gpu
                    pipe.set_outputs(images.gpu(), label.gpu())
                else:
                    pipe.set_outputs(images, label)

            pipe.build()
            last_batch_policy = LastBatchPolicy.DROP if training else LastBatchPolicy.PARTIAL
            super().__init__(pipe, reader_name="Reader", auto_reset=True, last_batch_policy=last_batch_policy)

        def __iter__(self):
            # if not reset (after an epoch), reset; if just initialize, ignore
            if self._counter >= self._size or self._size < 0:
                self.reset()
            return self

        def __next__(self):
            data = super().__next__()
            img, label = data[0]['data'], data[0]['label']
            img = (img - 127.5) / 127.5
            label = label.squeeze()
            if self.mixup_alpha > 0.0:
                if self.training:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    idx = torch.randperm(img.size(0)).to(img.device)
                    img = lam * img + (1 - lam) * img[idx, :]
                    label_a, label_b = label, label[idx]
                    lam = torch.tensor(lam, device=img.device, dtype=img.dtype)
                    label = {'targets_a': label_a, 'targets_b': label_b, 'lam': lam}
                else:
                    label = {
                        'targets_a': label,
                        'targets_b': label,
                        'lam': torch.ones((), device=img.device, dtype=img.dtype)
                    }
                return {'pixel_values': img, 'labels': label}
            return {'pixel_values': img, 'labels': label}

    rank = get_rank()
    world_size = get_world_size()

    train_pat = os.path.join(CONFIG['dataset'], 'train/*')
    train_idx_pat = os.path.join(CONFIG['dataset'], 'idx_files/train/*')
    val_pat = os.path.join(CONFIG['dataset'], 'validation/*')
    val_idx_pat = os.path.join(CONFIG['dataset'], 'idx_files/validation/*')

    train_data = DaliDataloader(sorted(glob.glob(train_pat)),
                                sorted(glob.glob(train_idx_pat)),
                                batch_size=CONFIG['hyperparameter']['batch_size'],
                                shard_id=rank,
                                num_shards=world_size,
                                gpu_aug=True,
                                cuda=True,
                                mixup_alpha=CONFIG['hyperparameter']['mixup_alpha'])

    test_data = DaliDataloader(sorted(glob.glob(val_pat)),
                               sorted(glob.glob(val_idx_pat)),
                               batch_size=CONFIG['hyperparameter']['batch_size'],
                               shard_id=rank,
                               num_shards=world_size,
                               training=False,
                               gpu_aug=False,
                               cuda=True,
                               mixup_alpha=CONFIG['hyperparameter']['mixup_alpha'])

    return train_data, test_data


def build_model():
    vit_config = ViTConfig(image_size=CONFIG['model']['img_size'],
                           patch_size=CONFIG['model']['patch_size'],
                           hidden_size=CONFIG['model']['hidden_size'],
                           intermediate_size=CONFIG['model']['intermediate_size'],
                           num_hidden_layers=CONFIG['model']['depth'],
                           hidden_dropout_prob=CONFIG['model']['dropout'],
                           num_attention_heads=CONFIG['model']['num_heads'],
                           num_labels=CONFIG['model']['num_labels'])
    model = ModelFromHF(vit_config, ViTForImageClassification)

    return model


class MixupLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, inputs, targets):
        targets_a, targets_b, lam = targets['targets_a'], targets['targets_b'], targets['lam']
        return lam * self.loss_fn(inputs, targets_a) + (1 - lam) * self.loss_fn(inputs, targets_b)


def build_loss():
    return MixupLoss()


def build_optimizer(params):
    optimizer = torch.optim.Adam(params,
                                 lr=CONFIG['hyperparameter']['learning_rate'],
                                 weight_decay=CONFIG['hyperparameter']['weight_decay'])

    return optimizer


def build_scheduler(epoch_steps, optimizer):
    from transformers.optimization import get_cosine_schedule_with_warmup

    max_steps = epoch_steps * CONFIG['hyperparameter']['num_epochs']
    warmup_steps = epoch_steps * CONFIG['hyperparameter']['warmup_epochs']
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warmup_steps,
                                                   num_training_steps=max_steps)

    return lr_scheduler


def vit_builder():
    model_type = CONFIG['model']['type']
    if model_type in _vit_configurations:
        for k, v in _vit_configurations[model_type].items():
            if k not in CONFIG['model']:
                CONFIG['model'][k] = v

    if 'hyperparameter' in CONFIG:
        for k, v in _default_hyperparameters.items():
            if k not in CONFIG['hyperparameter']:
                CONFIG['hyperparameter'][k] = v
    else:
        CONFIG['hyperparameter'] = _default_hyperparameters

    CONFIG['dataset'] = os.environ['DATA']

    return build_data, build_model, build_loss, build_optimizer, build_scheduler
