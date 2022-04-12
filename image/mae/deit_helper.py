from dataclasses import dataclass

# Interface code to call DeiT code under util/


@dataclass
class lr_sched_args:
    lr: float
    min_lr: float
    warmup_epochs: int = 40


@dataclass
class load_model_args:
    resume: str = ""
    start_epoch: int = 0
