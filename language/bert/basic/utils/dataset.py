import os
import random
import torch

from .vocab import Vocab
from .download import download_extract

SPLITTER = ' . '
NUM_DOWNLOAD_WORKERS = 4

# Get tokens of the BERT input sequence and their segment IDs
def _get_tokens_and_segments(tokens_a, tokens_b = None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens_b += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

# Generating the Next Sentence Prediction(nsp) Task
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i+1], paragraphs)
        # 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = _get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

# Generating the Masked Language Modeling(mlm) Task
# tokens: a list of tokens representing a BERT input sequence
# candidate_pred_positions: a list of token indices of the BERT input sequence excluding those of special tokens
# num_mlm_preds: the number of predictions (recall 15% random tokens to predict)
def _replace_mlm_tokens(tokens, candidata_pred_postions, num_mlm_preds, vocab):
    # The input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = tokens[:]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction
    random.shuffle(candidata_pred_postions)
    for mlm_pred_postion in candidata_pred_postions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_postion]
            # 10% replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_postion] = masked_token
        pred_positions_and_labels.append((mlm_pred_postion, tokens[mlm_pred_postion]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_postions = []
    for i, token in enumerate(tokens):
        # Ignore special tokens
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_postions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, 
        candidate_pred_postions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key = lambda x: x[0])
    pred_positions = [x[0] for x in pred_positions_and_labels]
    mlm_pred_labels = [x[1] for x in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

# Append special <mask> tokens into the inputs and padding them together
# exmaples: outputs from  _get_nsp_data_from_paragraph and _get_mlm_data_from_tokens
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # valid_lens excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

def _tokenize(lines, token='word'):
    # Split text lines into word or character tokens
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        paragraphs = [_tokenize(paragraph, token='word')
            for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
            for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5,
            reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        print("Get data for the next sentence prediction task...")
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, max_len))
        # Get data for the masked language model task
        print("Get data for the masked language model task...")
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]

        # Padding
        print("Padding inputs...")
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
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
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab