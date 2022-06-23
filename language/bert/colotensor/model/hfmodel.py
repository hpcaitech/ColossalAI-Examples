from colossalai.core import global_context as gpc
import torch

class ModelFromHF(torch.nn.Module):
    def __init__(self, config, model_cls):
        super().__init__()
        self.module = model_cls(config)
        if gpc.config.model.get('checkpoint'):
            self.module.apply(self.set_checkpointing)

    def set_checkpointing(self, module):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output.logits