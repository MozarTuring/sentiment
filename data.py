from torch.utils.data import Dataset
import pandas as pd
import sys
import time
import os


class fsaourDataset(Dataset):
  """Face Landmarks dataset."""

  def __init__(self, csv_file):
    self.fsaour = pd.read_csv(csv_file)

  def __len__(self):
    return len(self.fsaour)

  def __getitem__(self, idx):
    content = self.fsaour.iloc[idx, 1]
    label = self.fsaour.iloc[idx, 2:].tolist()
    sample = {'content': content, 'label': label}

    return sample


def string_process(string):
  string = string.replace('\n', ' ')
  string = string.replace('\r', ' ')
  string = string.replace('\t', ' ')
  string = string.strip('"')
  return string

def gendata(csv_file, i=None):
  # if csv_file == 'data/trainingset.csv':
  if csv_file == 'data/trainingset.csv':
    filename = 'data/char/' + 'ch_review' + str(i) + '/train.tsv'
  elif csv_file == 'data/validationset.csv':
    filename = 'data/char/' + 'ch_review' + str(i) + '/dev.tsv'
  elif csv_file == 'data/testset.csv':
    filename = 'data/char/test/test.tsv'
  else:
    print('error')
    sys.exit(1)

  fsdata = fsaourDataset(csv_file=csv_file)

  out = open(filename, 'w', encoding='utf8')
  start_time = time.time()
  for tt, dic in enumerate(fsdata):
    if tt % 10000 == 0:
      print(tt)
      print(time.time()-start_time)
    # out.write(' '.join(jieba.cut(string_process(dic['content']))) + '\t' + str(dic['label'][i]) + '\n')
    for st in string_process(dic['content']):
      out.write(st + ' ')

    out.write('\n')
    # out.write('\t' + str(dic['label'][i]) + '\n')

  out.close()
  # else:
  #   fsdata = fsaourDataset(csv_file=csv_file)
  #   filename1 = 'data/' + 'ch_review' + str(i) + '/dev.tsv'
  #   filename2 = 'data/' + 'ch_review' + str(i) + '/test.tsv'
  #   out1 = open(filename1, 'w', encoding='utf8')
  #   out2 = open(filename2, 'w', encoding='utf8')
  #   for dic in fsdata:
  #     out1.write(' '.join(jieba.cut(string_process(dic['content']))) + '\t' + str(dic['label'][i]) + '\n')
  #     out2.write(' '.join(jieba.cut(string_process(dic['content']))) + '\t' + str(dic['label'][i]) + '\n')
  #
  #   out1.close()
  #   out2.close()

if __name__ == '__main__':
  for i in range(20):
    if not os.path.exists('data/char/ch_review' + str(i)):
      os.makedirs('data/char/ch_review' + str(i))
    gendata(csv_file='data/trainingset.csv', i=i)
    gendata(csv_file='data/validationset.csv', i=i)

  gendata(csv_file='data/testset.csv')



