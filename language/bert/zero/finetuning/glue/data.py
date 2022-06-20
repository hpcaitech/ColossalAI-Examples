import torch
import os
import pickle
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from torch.utils.data import TensorDataset
from processors import convert_examples_to_features


def gen_tensor_dataset(features):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long,
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in features],
        dtype=torch.long,
    )
    return TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )


def get_train_features(data_dir, vocab_file, max_seq_length, do_lower_case, tokenizer, processor):

    cached_train_features_file = os.path.join(
        data_dir,
        '{0}_{1}_{2}'.format(
            vocab_file,
            str(max_seq_length),
            str(do_lower_case),
        ),
    )
    train_features = None
    logger = get_dist_logger()
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
        logger.info("Loaded pre-processed features from {}".format(cached_train_features_file))
    except:
        logger.info("Did not find pre-processed features from {}".format(cached_train_features_file))
        train_examples = processor.get_train_examples(data_dir)
        train_features, _ = convert_examples_to_features(
            train_examples,
            processor.get_labels(),
            max_seq_length,
            tokenizer,
        )
        if gpc.get_global_rank() == 0:
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
    return train_features
