import torch
from torch import nn
from utils.dataset import load_data_wiki
from utils.bert import BERTModel
'''Profiling'''
from utils.profile import Timer, Accumulator
'''Profiling'''

# BERT Loss for mlm and nsp
def get_loss_bert(model, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_Y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = model(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)

    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)

    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_Y)

    # BERT loss is the sum of mlm loss and nsp losss
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

def train(train_iter, model, loss, optim, vocab_size, devices, num_steps):
    model = nn.DataParallel(model, device_ids = devices).to(devices[0])
    trainer = optim(model.parameters(), lr = 0.01)
    step = 0
    num_steps_reached = False

    '''Profiling'''
    timer = Timer()
    metric = Accumulator(4)
    '''Profiling'''

    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_Y in train_iter:
            # Move to the same device
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y = mlm_Y.to(devices[0])
            nsp_Y = nsp_Y.to(devices[0])

            # train
            trainer.zero_grad()
            '''Profiling'''
            timer.start()
            '''Profiling'''
            mlm_l, nsp_l, l = get_loss_bert(model, loss, vocab_size, tokens_X, segments_X,
                                            valid_lens_x, pred_positions_X, mlm_weights_X,
                                            mlm_Y, nsp_Y)
            l.backward()
            trainer.step()
            '''Profiling'''
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            '''Profiling'''
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}, '
          f'avg secs: {timer.avg()}')

def main():
    dataset_name = 'wikitext-103'
    batch_size = 512
    max_len = 64

    # DataLoader
    train_iter, vocab = load_data_wiki(dataset_name, batch_size, max_len)
    model = BERTModel(len(vocab), num_hiddens = 128, norm_shape = [128],
                      ffn_num_input = 128, ffn_num_hiddens = 256, num_heads = 2,
                      num_layers = 2, dropout = 0.2, key_size = 128, query_size = 128,
                      value_size = 128, hid_in_features = 128, mlm_in_features = 128,
                      nsp_in_features = 128)

    # Devices
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    if not devices:
        devices = torch.device('cpi')

    # Loss
    loss = nn.CrossEntropyLoss()

    # Optim
    optim = torch.optim.Adam

    # Train
    train(train_iter, model, loss, optim, len(vocab), devices, 50)

if __name__ == '__main__':
    main()
