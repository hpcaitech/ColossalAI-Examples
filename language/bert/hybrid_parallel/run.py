import os

from bert.common.helper import bert_builder
from bert.colossalai_utils.utils import init_w_col
from bert.common.train import train
from zero.common.utils import CONFIG, load_config, print_log
from zero.torch_utils.utils import init_w_torch

_method = {
    'torch': init_w_torch,
    'colossalai': init_w_col,
}

_builder = {
    'bert': bert_builder,
}


def run_bert():
    method = CONFIG['method']

    model = CONFIG['model']['type']
    model_type = model.split('_')[0]

    train(*_method[method](_builder[model_type]))


if __name__ == '__main__':
    load_config()

    CONFIG['log_path'] = os.environ.get('LOG', '.')
    os.makedirs(CONFIG['log_path'], exist_ok=True)

    print_log(f'Initializing {CONFIG["method"]} ...')

    run_bert()
