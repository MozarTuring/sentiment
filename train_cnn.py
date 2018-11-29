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

args = get_args_cnn('train')
TD = args.TASK_ID
USE_GPU, DEVICE = get_cuda(args)

# if isinstance(args.filter_sizes, list):
#   args.filter_sizes = ''.join(list(map(str, args.filter_sizes)))
out_dir = make_out_dir(args)
writer = get_writer(args, out_dir)
logger = get_logger(out_dir)

split_tool = 'char'
dic_data, text_fields, label_field = get_fields('data/'+split_tool+'/', args.TASK_ID, 'train')
embed_file = split_tool + '_embed3'
# embed_file = None
pre_embedding, vocab_ls, vocab_dic = load_embeddings(args.EMBEDDING_DIM, text_fields, logger, emb_file=embed_file)

text_fields['train'].vocab.itos = vocab_ls
text_fields['train'].vocab.stoi = vocab_dic
text_fields['dev'].vocab.itos = vocab_ls
text_fields['dev'].vocab.stoi = vocab_dic

train_iter, dev_iter = \
    data.BucketIterator.splits((dic_data['train'], dic_data['dev']), batch_sizes=(args.BATCH_SIZE, args.BATCH_SIZE),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)


weights = get_weights(train_iter)
logger.info('{}'.format(weights))

if args.model == 'cnn':
  get_filter_sizes(args)

model = create_model(args, vocab_ls, label_field, USE_GPU, DEVICE, logger)
if pre_embedding:
  model.embeddings.weight.data.copy_(torch.as_tensor(pre_embedding))
model = load_model(out_dir, model, logger)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
loss_functions = [nn.NLLLoss(weight=weight).to(DEVICE) for weight in weights]

logger.info('Start training')
GLOBAL_STEP = get_GLOBAL_STEP(out_dir)
logger.info('Global step is {}'.format(GLOBAL_STEP))

train_start(args.EPOCHS, model, train_iter, dev_iter, loss_functions,
            optimizer, writer, logger, out_dir, GLOBAL_STEP)

writer.close()

# if __name__ == '__main__':
#   for i in range(1):
#     args.TASK_ID = str(int(TD) + i)
#     loop()