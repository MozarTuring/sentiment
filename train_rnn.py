from torch import optim
import json
import random
import os
import torch.nn as nn
import sys
import argparse
import ipdb
from utils import *
from vocab_embed import load_embeddings
from get_args import *
from create_model import create_model

# torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

args = get_args_rnn('train')
TD = args.TASK_ID
USE_GPU, DEVICE = get_cuda(args)
NUM_ELE = 26

# if isinstance(args.filter_sizes, list):
#   args.filter_sizes = ''.join(list(map(str, args.filter_sizes)))
out_dir = make_out_dir(args, NUM_ELE)
writer = get_writer(args, out_dir)
logger = get_logger(out_dir)

split_tool = 'sentence'
dic_data, text_fields, label_field = get_fields('data/' + split_tool + '/',
                                                args.TASK_ID, 'train')
embed_file = split_tool + '_embed3'
# embed_file = None
pre_embedding, vocab_ls, vocab_dic = \
  load_embeddings(args.EMBEDDING_DIM, text_fields, logger, emb_file=embed_file)

change_vocab(text_fields['train'], vocab_dic, vocab_ls)
change_vocab(text_fields['dev'], vocab_dic, vocab_ls)

if NUM_ELE == 6:
    vocab_dicb = {'<unk>': 0, '<pad>': 1}
    for lb in label_field.vocab.itos[2:]:
        vocab_dicb[lb] = sgn(int(lb)) + 3
    vocab_lb = ['<unk>', '<pad>', -1, 0, 1]

    change_vocab(label_field, vocab_dicb, vocab_lb)

# for dtrain in dic_data['train']:
#   text_fields['train'].process([dtrain.text])
#   print(dtrain)

train_iter, dev_iter = \
    data.BucketIterator.splits((dic_data['train'], dic_data['dev']),
                               batch_sizes=(args.BATCH_SIZE, args.BATCH_SIZE),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)

if NUM_ELE == 26:
    weights = get_weights(train_iter, label_field, vocab_ls)
else:
    weights = get_weights(train_iter, NUM_ELE, len(label_field.vocab) - 2)
logger.info('{}'.format(weights))

if args.model == 'cnn':
    get_filter_sizes(args)

model = create_model(args, vocab_ls, label_field, USE_GPU, DEVICE, NUM_ELE,
                     logger)
if pre_embedding:
    model.embeddings.weight.data.copy_(torch.as_tensor(pre_embedding))
model = load_model(out_dir, model, logger)

optimizer = optim.Adam(model.parameters(), lr=1e-3)  # weight_decay=0.00001)
loss_functions = [nn.NLLLoss(weight=weight).to(DEVICE) for weight in weights[0]]\
                 + [nn.NLLLoss(weight=weight).to(DEVICE) for weight in weights[1]]

logger.info('Start training')
GLOBAL_STEP = get_GLOBAL_STEP(out_dir)
logger.info('Global step is {}'.format(GLOBAL_STEP))

train_start(args.EPOCHS, model, train_iter, dev_iter, loss_functions,
            optimizer, writer, logger, out_dir, label_field, DEVICE,
            GLOBAL_STEP)

writer.close()
