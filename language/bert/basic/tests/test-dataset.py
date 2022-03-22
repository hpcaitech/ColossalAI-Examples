import sys
import os
import_path = os.path.abspath('./')
sys.path.append(import_path)
print(import_path)

from utils import load_data_wiki

dataset_name = 'wikitext-103'
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(dataset_name, batch_size, max_len)

print(len(vocab))

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break