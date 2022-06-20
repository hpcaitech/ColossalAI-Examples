import colossalai
from processors import PROCESSORS

__all__ = ['parse_args']


def parse_args():
    parser = colossalai.get_default_parser()
    add_bert_args(parser)
    add_training_args(parser)
    add_control_args(parser)
    add_data_args(parser)
    add_eval_args(parser)
    return parser.parse_args()


def add_bert_args(parser):
    group = parser.add_argument_group('bert model configs')
    group.add_argument("--bert_config", type=str, required=True, help="The BERT model config")
    group.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")


def add_training_args(parser):
    group = parser.add_argument_group('training configs')
    group.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints "
        "will be written.",
    )
    group.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        required=True,
        help="The checkpoint file from pretraining",
    )
    group.add_argument("--train_batch_size", default=16, type=int, help="Batch size per GPU for training.")
    group.add_argument("--learning_rate", default=2.4e-5, type=float, help="The initial learning rate for Adam.")
    group.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    group.add_argument("--max_steps", default=-1, type=int, help="Total number of training steps to perform.")
    group.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup "
        "for. E.g., 0.1 = 10%% of training.",
    )
    group.add_argument('--skip_checkpoint', default=False, action='store_true', help="Whether to save checkpoints")


def add_eval_args(parser):
    group = parser.add_argument_group('evaluation configs')
    group.add_argument("--eval_batch_size", default=16, type=int, help="Batch size per GPU for eval.")


def add_data_args(parser):
    group = parser.add_argument_group('data configs')
    group.add_argument('--vocab_file',
                       type=str,
                       default=None,
                       required=True,
                       help="Vocabulary mapping/file BERT was pretrainined on")
    group.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data "
        "files) for the task.",
    )
    group.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        choices=PROCESSORS.keys(),
        help="The name of the task to train.",
    )
    group.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece "
        "tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )


def add_control_args(parser):
    group = parser.add_argument_group('control configs')
    group.add_argument('--train', action='store_true', help="Run training")
    group.add_argument('--eval', action='store_true', help="Run evaluation")
    group.add_argument('--predict', action='store_true', help="Run prediction")
