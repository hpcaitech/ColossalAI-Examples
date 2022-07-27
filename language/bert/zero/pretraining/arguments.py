import colossalai
from numpy import require

__all__ = ['parse_args']


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('--bert-config', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--warmup-ratio', default=0.01, type=float)
    parser.add_argument('--vocab-file', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--save-checkpoint-interval', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    return args
