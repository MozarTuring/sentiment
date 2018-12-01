from torch.utils.data import Dataset
import pandas as pd
import sys
import time
from utils import string_process, SEP
import os
import jieba
# from pyhanlp import *


jieba.add_word(SEP)
class fsaourDataset(Dataset):
  """Face Landmarks dataset."""

  def __init__(self, csv_file):
    self.fsaour = pd.read_csv(csv_file, header=0, encoding='utf8')

  def __len__(self):
    return len(self.fsaour)

  def __getitem__(self, idx):
    content = self.fsaour.iloc[idx, 1]
    label = self.fsaour.iloc[idx, 2:].tolist()
    sample = {'content': content, 'label': label}

    return sample


def gendata(csv_file, path, sub, split_tool=None, i=None):
  # if csv_file == 'data/trainingset.csv':
  if csv_file == 'data/trainingset.csv':
    if i:
      filename = path + 'ch_review' + str(i) + '/train.tsv'
    else:
      filename = path + sub + '/train.tsv'
  elif csv_file == 'data/validationset.csv':
    if i:
      filename = path + 'ch_review' + str(i) + '/dev.tsv'
    else:
      filename = path + sub + '/dev.tsv'
  elif csv_file == 'data/testset.csv':
    filename = path + 'test/test.tsv'
  elif csv_file == 'data/testsetb.csv':
    filename = path + 'testb/test.tsv'
  else:
    print('error')
    sys.exit(1)

  fsdata = fsaourDataset(csv_file=csv_file)

  out = open(filename, 'w', encoding='utf8')
  start_time = time.time()
  for tt, dic in enumerate(fsdata):
    if (tt+1) % 10000 == 0:
      print(tt)
      print(time.time()-start_time)

    if split_tool == 'jieba':
      if i == None:
        lss = [sum(dic['label'][:3]), sum(dic['label'][3:7]), sum(dic['label'][7:10]), sum(dic['label'][10:14]),
               sum(dic['label'][14:18]), sum(dic['label'][18:])]
        out.write(' '.join(jieba.cut(string_process(dic['content']))) + '\t' + ' '.join(map(str, lss)) + '\n')
      else:
        out.write(' '.join(jieba.cut(string_process(dic['content']))) + '\t' + ' '.join(map(str, dic['label'])) + '\n')
    elif split_tool == 'hanlp':
      ls = []
      for term in HanLP.segment(string_process(dic['content'])):
        ls.append(term.word)
      if i==None:
        out.write(' '.join(ls) + '\n')
      else:
        out.write(' '.join(ls) + '\t' + str(dic['label'][i]) + '\n')
    elif split_tool == 'char':
      for st in string_process(dic['content']):
        out.write(st + ' ')
      out.write('\n') # '\t' + ' '.join(map(str, dic['label'])) +
    elif split_tool == 'sentence':
      # if set(['味道','不错','很','划算']).issubset(dic['content'].split()) and len(dic['content'].split()) < 50:
      #   import ipdb
      #   ipdb.set_trace()
      out.write(' '.join(jieba.cut(string_process(dic['content']))) + '\t' + ' '.join(map(str, dic['label'])) + '\n')

  out.close()

if __name__ == '__main__':
  # path = 'data/token2/'
  # for i in range(1):
  #   if not os.path.exists(path+'ch_review' + str(i)):
  #     os.makedirs(path+'ch_review' + str(i))
  #   gendata('data/trainingset.csv', 'token', path, 'hanlp', i=i)
  #   gendata('data/validationset.csv', 'token', path, 'hanlp', i=i)
  # gendata('data/testset.csv', 'token', path, 'hanlp',)

  split_tool = 'sentence'
  path = 'data/' + split_tool + '/'
  sub = 'fine'
  gendata('data/testsetb.csv', path, sub, split_tool)
  if not os.path.exists(path + sub):
    os.makedirs(path + sub)
  gendata('data/trainingset.csv', path, sub, split_tool)
  gendata('data/validationset.csv', path, sub, split_tool)




