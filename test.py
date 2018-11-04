from utils import string_process, get_fields, load_embeddings
from torchtext import data
import random
import pandas as pd
import argparse
import jieba
import ipdb
from utils import make_out_dir, get_Model_Name
from create_model import create_model
from models import *


# ipdb.set_trace()
torch.manual_seed(1)
random.seed(1)

args = argparse.ArgumentParser()
args.add_argument('--cu', dest='CUDA_NUM', type=str)
args.add_argument('--m', dest='model', help='specify the mode to use (default: lstm)')
args.add_argument('--attn', dest='attn_method', help='general, dot, concat')
args.add_argument('--emdim', dest='EMBEDDING_DIM', type=int)
args.add_argument('--hdim', dest='HIDDEN_DIM', type=int)
args.add_argument('--bsize', dest='BATCH_SIZE', type=int)
args.add_argument('--p', dest='paral', default=False, type=bool)
args.add_argument('--id', dest='TASK_ID', default=0, type=str)
args = args.parse_args()

USE_GPU = torch.cuda.is_available()
cuda_str = 'cuda:' + args.CUDA_NUM
DEVICE = torch.device(cuda_str if USE_GPU else "cpu")

dic_data, text_fields, label_field = get_fields()
pre_embedding, vocab_ls, vocab_dic = load_embeddings(args, text_fields, emb_file='new_embed')

text_fields['test'].vocab.itos = vocab_ls
text_fields['test'].vocab.stoi = vocab_dic

test_iter, = \
    data.BucketIterator.splits((dic_data['test'], ), batch_sizes=(args.BATCH_SIZE, ),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)

DISCARD = [' ', '\xa0', '\u3000', '\u200a']
LABELS = ['-2', '-1', '0', '1']
df = pd.read_csv('data/testset.csv', header=0, encoding='utf-8')
ls_senten = [string_process(df.iloc[row, 1]) for row in range(len(df))]

ls_check = [[vocab_dic[st.lower()] for st in jieba.cut(senten) if st not in DISCARD] for senten in ls_senten]
ls_sent_len = [len(se) for se in ls_check]
print('Maximum length of testset is {}\n'.format(max(ls_sent_len)))

model = create_model(args, vocab_ls, label_field, USE_GPU, DEVICE)

for i in range(20):
  print('Start testing for task {}'.format(i))
  args.TASK_ID = str(i)
  out_dir = make_out_dir(args)
  print('Loading model from {}'.format(out_dir))
  model.load_state_dict(torch.load(out_dir + '/best_model.pth'))
  print('Complete loading')

  for num_iter, batch in enumerate(test_iter):
    sent = batch.text
    pred = model(torch.transpose(sent, 0, 1))
    pred_label = pred.data.max(1)[1]
    ls_pred = pred_label.tolist()
    sent_ls = torch.transpose(sent, 0, 1).tolist()
    unpad_sent_ls = [[sen for sen in sent if sen != 1] for sent in sent_ls]
    # ls_sentence = [''.join([idx2word[it] for it in sent_ls[itt] if it != 1]) for itt in range(len(sent_ls))]
    for tt, unpad_sent in enumerate(unpad_sent_ls):
      if unpad_sent not in ls_check:
        ipdb.set_trace()
        print('hhhh')
      df.iloc[ls_check.index(unpad_sent), i+2] = label_field.vocab.itos[ls_pred[tt] + 1]

  for roww in range(len(df)):
    if df.iloc[roww, i+2] not in LABELS:
      ipdb.set_trace()

  print('Complete testing for task {} in {} iterations\n'.format(i, num_iter))

result_path = 'results/' + get_Model_Name(args) + '.csv'
print('Writing result to {}'.format(result_path))
df.to_csv(result_path, encoding='utf_8_sig', index=False)
