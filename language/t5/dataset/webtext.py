import json
import os
import random

import torch
from colossalai.registry import DATASETS
from torch._C import DictType
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from colossalai.logging import get_dist_logger


def racha_detection(lista):
    # It returns a list of lists where each sub-list contains the consecutive tokens in the list
    rachas = []
    racha = []
    for i, element in enumerate(lista):
        if (i < len(lista) - 1) and (lista[i + 1] == element + 1):
            racha.append(element)
        else:
            if len(racha) > 0:
                rachas.append(racha + [element])
            else:  # (i!=len(lista)-1):
                rachas.append([element])
            racha = []
    return rachas


def masking(tokenized_sentence, rachas, tokenizer, seq_len):
    # Function to mask a tokenized_sentence (token ids) following the rachas described in rachas
    # Only one sentinel_token per racha
    sent_token_id = 0
    enmascared = tokenized_sentence.copy()
    for racha in rachas:
        sent_token = f'<extra_id_{sent_token_id}>'
        sent_id = tokenizer.encode(sent_token)[0]
        for i, idx in enumerate(racha):
            if i == 0:
                enmascared[idx] = sent_id
            else:
                enmascared[idx] = -100
        sent_token_id += 1
    enmascared = [t for t in enmascared if t != -100]
    enmascared.extend([tokenizer.pad_token_id]*(seq_len - len(enmascared)))

    return enmascared


def add_noise(tokenized_sentence, tokenizer, seq_len, percent=0.15):
    # Function that takes a sentence, tokenizer and a noise percentage and returns
    # the masked input_ids and masked target_ids according to the T5 paper and HuggingFace docs
    # To see the process working uncomment all the prints ;)
    #inputs = []
    #labels = []
    #for tokenized_sentence in tokenized_sentences:
        #tokenized_sentence = tokenizer.encode(sentence)
        #print('PRE-MASKED:')
        #print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(tokenized_sentence)))

    idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)),
                                       int(len(tokenized_sentence)*percent)))
    rachas = racha_detection(idxs_2_mask)
    enmascared_input = masking(tokenized_sentence, rachas, tokenizer, seq_len)
    #print('RACHAS INPUT: {}'.format(rachas))
    idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
    rachas = racha_detection(idxs_2_mask)
    enmascared_target = masking(tokenized_sentence, rachas, tokenizer, seq_len)
    #print('RACHAS TARGET: {}'.format(rachas))
    #inputs.append(enmascared_input)
    #labels.append(enmascared_target)

    #print('POST-MASKED:')
    #print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_input)))
    #print('TARGET: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_target)))

    #inputs = torch.LongTensor(inputs)
    #labels = torch.LongTensor(labels)
    return torch.LongTensor(enmascared_input), torch.LongTensor(enmascared_target)


@DATASETS.register_module
class WebtextDataset(Dataset):
    def __init__(self, path, seq_len=1024) -> None:
        super().__init__()
        root = os.path.dirname(path)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.tokenizer.decoder_start_token_id = self.tokenizer.pad_token_id
        encoded_data_cache_path = os.path.join(root, f'gpt_webtext_{seq_len}.pt')
        if os.path.isfile(encoded_data_cache_path):
            seq_len_, data = torch.load(encoded_data_cache_path)
            if seq_len_ == seq_len:
                self.data = data
                self.seq_len = seq_len_
                #self.labels = labels
                return
        self.seq_len = seq_len
        with open(path) as f:
          #for line in f.readlines():
          raw_data = [json.loads(l)['text'] for l in f.readlines()]
          #for line in f.readlines():
          #    raw_data.append(json.loads(line)['text'])
        #tokenizer.pad_token = tokenizer.unk_token
        self.data = self.tokenizer(raw_data, padding=True, truncation=True, max_length=seq_len)['input_ids']
        #self.data = encoded_data['input_ids']
        #self.attention_mask = encoded_data['attention_mask']
        #self.inputs, self.labels = add_noise(encoded_data['input_ids'], tokenizer, seq_len)
        torch.save((seq_len, self.data), encoded_data_cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, labels = add_noise(self.data[index], self.tokenizer, self.seq_len)
        return {'input_ids': input_ids,
            'labels': labels}, torch.LongTensor(self.data[index])

if __name__ == '__main__':
  #wt = WebtextDataset('./real_data/shuffled_deduplicated_urls.json', 4)
  raw_data = "What are your plans for today little miss i don't want to do anything for now"
  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  tokenizer.pad_token = tokenizer.unk_token
  encoded_data = tokenizer(raw_data, padding=True, truncation=True, max_length=512)
  inputs, labels = add_noise(encoded_data['input_ids'], tokenizer, 512)
  print(tokenizer.decode(inputs, skip_special_tokens=True))
  print(tokenizer.decode(labels, skip_special_tokens=True))