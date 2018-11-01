from utils import string_process
from utils import load_review_test
from torchtext import data
import torch
import pandas as pd
import argparse
from train import BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM
import jieba
from models import *

args = argparse.ArgumentParser()
args.add_argument('--m', dest='model', default='bilstm', help='specify the mode to use (default: lstm)')
args = args.parse_args()

DISCARD = [' ', '\xa0', '\u3000', '\u200a']
df = pd.read_csv('data/testset.csv', header=0, encoding='utf-8')
ls_senten = [string_process(df.iloc[row, 1]) for row in range(len(df))]

USE_GPU = torch.cuda.is_available()
print('testing....')

for i in range(20):
  cuda_str = "cuda:" + str(i % 8)
  DEVICE = torch.device(cuda_str if torch.cuda.is_available() else "cpu")

  text_field = data.Field(lower=True)
  label_field = data.Field(sequential=False)

  train_iter, dev_iter, test_iter = load_review_test(text_field, label_field, BATCH_SIZE)

  idx2word = text_field.vocab.itos
  word2idx = text_field.vocab.stoi

  ls_check = [[word2idx[st.lower()] for st in jieba.cut(senten) if st not in DISCARD] for senten in ls_senten]

  if args.model == 'lstm':
      model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                      vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,
                      use_gpu=USE_GPU, batch_size=BATCH_SIZE, devic=DEVICE)

  if args.model == 'bilstm':
      model = BiLSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                            use_gpu=USE_GPU, batch_size=BATCH_SIZE)

  if args.model == 'gru':
    embedding = nn.Embedding(len(text_field.vocab), EMBEDDING_DIM)
    model = GRUSentiment(hidden_size=HIDDEN_DIM, embedding=embedding, n_layers=1, dropout=0)

  if args.paral:
    model = nn.DataParallel(model)

  model.to(DEVICE)
  model.load_state_dict(torch.load('model' + str(i) + '/best_model.pth'))
  for batch in test_iter:
    sent, label = batch.text, batch.label
    model.batch_size = len(label.data)
    pred = model(torch.transpose(sent, 0, 1))
    pred_label = pred.data.max(1)[1]
    ls_pred = pred_label.tolist()
    sent_ls = torch.transpose(sent, 0, 1).tolist()
    unpad_sent_ls = [[sen for sen in sent if sen != 1] for sent in sent_ls]
    # ls_sentence = [''.join([idx2word[it] for it in sent_ls[itt] if it != 1]) for itt in range(len(sent_ls))]
    for tt, unpad_sent in enumerate(unpad_sent_ls):
      if unpad_sent not in ls_check:
        import ipdb
        ipdb.set_trace()
        print('hhhh')
      df.iloc[ls_check.index(unpad_sent), i+2] = label_field.vocab.itos[ls_pred[tt] + 1]

df.to_csv('data/test/result.csv', encoding='utf_8_sig', index=False)