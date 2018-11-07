from torch import optim
import json
import random
import os
import torch.nn as nn
import sys
import ipdb
from utils import *
from vocab_embed import load_embeddings
from get_args import *
from create_model import create_model
from tensorboardX import SummaryWriter


# torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

args = get_args_cnn()

USE_GPU, DEVICE = get_cuda(args)

out_dir = make_out_dir(args)
writer = SummaryWriter(log_dir=out_dir)
logger = get_logger(out_dir)

dic_data, text_fields, label_field = get_fields(args.TASK_ID)
new_embed_file = 'char_embed'
# filter_embed_txt('embed', text_fields, new_embed_file)
# sys.exit()
pre_embedding, vocab_ls, vocab_dic = load_embeddings(args, text_fields, logger, emb_file=new_embed_file)

text_fields['train'].vocab.itos = vocab_ls
text_fields['train'].vocab.stoi = vocab_dic
text_fields['dev'].vocab.itos = vocab_ls
text_fields['dev'].vocab.stoi = vocab_dic

train_iter, dev_iter = \
    data.BucketIterator.splits((dic_data['train'], dic_data['dev']), batch_sizes=(args.BATCH_SIZE, args.BATCH_SIZE),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)

weight = get_weight(train_iter)
logger.info('{}'.format(weight))

get_filter_sizes(args)

model = create_model(args, vocab_ls, label_field, logger, USE_GPU, DEVICE)
model.embeddings.weight.data.copy_(torch.as_tensor(pre_embedding))
model = load_model(out_dir, model, logger)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.NLLLoss(weight=weight.to(DEVICE)).to(DEVICE)

logger.info('Start training')
GLOBAL_STEP = get_GLOBAL_STEP(out_dir)

train_start(args.EPOCHS, model, train_iter, dev_iter, loss_function, optimizer, writer, logger, out_dir, GLOBAL_STEP)

writer.close()
