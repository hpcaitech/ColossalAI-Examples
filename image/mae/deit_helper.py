from dataclasses import dataclass

# Interface code to call DeiT code under util/


@dataclass
class load_model_args:
    resume: str
    start_epoch: int
