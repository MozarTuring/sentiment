import torch
import numpy as np
from torchtext import data


def string_process(string):
  string = string.replace('\n', ' ')
  string = string.replace('\r', ' ')
  string = string.replace('\t', ' ')
  # string = string.replace(' ', '')
  string = string.strip('"')
  return string


def load_bin_vec(fname, vocab):
  """
  Loads 300x1 word vecs from Google (Mikolov) word2vec
  """
  word_vecs = {}
  with open(fname, "rb") as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in range(vocab_size):
      word = []
      while True:
        ch = f.read(1).decode('latin-1')
        if ch == ' ':
          word = ''.join(word)
          break
        if ch != '\n':
          word.append(ch)
      if word in vocab:
        word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
      else:
        f.read(binary_len)
  return word_vecs


def load_review_train(text_field, label_field, batch_size, taskid, DEVICE):
  train, dev, test = data.TabularDataset.splits(path='data/ch_review'+taskid+'/', train='train.tsv',
                                                validation='dev.tsv', test='test.tsv', format='tsv',
                                                fields=[('text', text_field), ('label', label_field)])

  text_field.build_vocab(train, dev, test)
  label_field.build_vocab(train, dev, test)

  train_iter, dev_iter, test_iter = \
    data.BucketIterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)

  return train_iter, dev_iter, test_iter


def load_review_test(text_field, label_field, batch_size, DEVICE):
  train, dev, test = data.TabularDataset.splits(path='data/test/', train='train.tsv',
                                                validation='dev.tsv', test='dev.tsv', format='tsv',
                                                fields=[('text', text_field), ('label', label_field)])

  text_field.build_vocab(train, dev, test)
  label_field.build_vocab(train, dev, test)

  train, dev, test = data.TabularDataset.splits(path='data/test/', train='train.tsv',
                                                validation='dev.tsv', test='test.tsv', format='tsv',
                                                fields=[('text', text_field), ('label', label_field)])

  train_iter, dev_iter, test_iter = \
    data.BucketIterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)

  return train_iter, dev_iter, test_iter


def get_accuracy(truth, pred):
  assert len(truth) == len(pred)
  right = 0
  for i in range(len(truth)):
    if truth[i] == pred[i]:
      right += 1.0
  return right / len(truth)

def evaluate(model, data, loss_function, name):
  model.eval()
  avg_loss = 0.0
  truth_res = []
  pred_res = []
  for batch in data:
    sent, label = batch.text, batch.label
    label.data.sub_(1)
    truth_res += list(label.data)
    model.batch_size = len(label.data)
    pred = model(torch.transpose(sent, 0, 1))
    pred_label = pred.data.max(1)[1]
    pred_res += [x for x in pred_label]
    loss = loss_function(pred, label)
    avg_loss += loss.data
  avg_loss /= len(data)
  acc = get_accuracy(truth_res, pred_res)
  print(name + ': loss %.2f acc %.1f' % (avg_loss, acc * 100))
  return acc


# def train_epoch(model, train_iter, loss_function, optimizer):
#   model.train()
#   avg_loss = 0.0
#   truth_res = []
#   pred_res = []
#   count = 0
#   for batch in train_iter:
#     sent, label = batch.text, batch.label
#     label.data.sub_(1)
#     truth_res += list(label.data)
#     model.batch_size = len(label.data)
#     pred = model(sent)
#     pred_label = pred.data.max(1)[1].numpy()
#     pred_res += [x for x in pred_label]
#     model.zero_grad()
#     loss = loss_function(pred, label)
#     avg_loss += loss.data[0]
#     count += 1
#     loss.backward()
#     optimizer.step()
#   avg_loss /= len(train_iter)
#   acc = get_accuracy(truth_res, pred_res)
#   return avg_loss, acc