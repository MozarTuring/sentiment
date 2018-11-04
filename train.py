from torch import optim
import random
import os
import torch.nn as nn
import torch
from torchtext import data
import argparse
import ipdb
from utils import get_fields, train_start, make_out_dir, load_embeddings
from create_model import create_model


# ipdb.set_trace()
# torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

args = argparse.ArgumentParser()
args.add_argument('--cu', dest='CUDA_NUM', type=str)
args.add_argument('--id', dest='TASK_ID', type=str)
args.add_argument('--m', dest='model', help='specify the mode to use (default: lstm)')
args.add_argument('--attn', dest='attn_method', help='general, dot, concat')
args.add_argument('--emdim', dest='EMBEDDING_DIM', type=int)
args.add_argument('--hdim', dest='HIDDEN_DIM', type=int)
args.add_argument('--bsize', dest='BATCH_SIZE', type=int)
args.add_argument('--epos', dest='EPOCHS', type=int)
args.add_argument('--p', dest='paral', default=False, type=bool)
args = args.parse_args()

USE_GPU = torch.cuda.is_available()
cuda_str = 'cuda:' + args.CUDA_NUM
DEVICE = torch.device(cuda_str if USE_GPU else "cpu")

out_dir = make_out_dir(args)
log_name = out_dir + '/log'
log_file = open(log_name, 'w+', encoding='utf8')
log_file.write('log will be saved to {}\n'.format(log_name))
log_file.write("model will be saved to {}\n".format(out_dir))

dic_data, text_fields, label_field = get_fields(args.TASK_ID)
pre_embedding, vocab_ls, vocab_dic = load_embeddings(args, text_fields, emb_file='new_embed')

text_fields['train'].vocab.itos = vocab_ls
text_fields['train'].vocab.stoi = vocab_dic
text_fields['dev'].vocab.itos = vocab_ls
text_fields['dev'].vocab.stoi = vocab_dic

train_iter, dev_iter = \
    data.BucketIterator.splits((dic_data['train'], dic_data['dev']), batch_sizes=(args.BATCH_SIZE, args.BATCH_SIZE),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)
dic_iter = {'train': train_iter, 'dev': dev_iter}

model = create_model(args, vocab_ls, label_field, USE_GPU, DEVICE)
model.embeddings.weight.data.copy_(torch.as_tensor(pre_embedding))

if os.path.exists(out_dir + '/best_model.pth'):
  model.load_state_dict(torch.load(out_dir + '/best_model.pth'))
  log_file.write('Loaded model as the initial model\n')

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()

print('Start training')
train_start(args.EPOCHS, model, dic_iter, loss_function, optimizer, log_file, out_dir)

log_file.close()

