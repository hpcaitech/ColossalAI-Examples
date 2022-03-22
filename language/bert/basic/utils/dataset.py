import os
import random
import torch

from .vocab import Vocab
from .download import download_extract
from .tokenizer import tokenize, get_nsp_data_from_paragraph, get_mlm_data_from_tokens, pad_bert_inputs

SPLITTER = ' . '
NUM_DOWNLOAD_WORKERS = 4

class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        paragraphs = [tokenize(paragraph, token='word')
            for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
            for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5,
            reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        print("Get data for the next sentence prediction task...")
        examples = []
        for paragraph in paragraphs:
            examples.extend(get_nsp_data_from_paragraph(paragraph, paragraphs, max_len))
        # Get data for the masked language model task
        print("Get data for the masked language model task...")
        examples = [(get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]

        # Padding
        print("Padding inputs...")
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = pad_bert_inputs(
            examples, max_len, self.vocab)
        
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])
    
    def __len__(self):
        return len(self.all_token_ids)

def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # To lowercase
    paragraphs = [line.strip().lower().split(SPLITTER)
                    for line in lines if len(line.split(SPLITTER)) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

def load_data_wiki(dataset_name, batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = NUM_DOWNLOAD_WORKERS
    data_dir = download_extract(dataset_name, dataset_name)
    paragraphs = _read_wiki(data_dir)
    train_set = WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab