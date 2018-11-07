import torch
import json
import numpy as np
from torchtext import data
import os
from sklearn import metrics
import logging
import sys


def load_model(out_dir, model, logger):
  if os.path.exists(out_dir + '/best_model.pth'):
    model.load_state_dict(torch.load(out_dir + '/best_model.pth'))
    logger.info('Initial model is loaded')
  else:
    logger.info('Initial model is fresh')

  return model


def get_GLOBAL_STEP(out_dir):
  if os.path.exists(out_dir + '/global_step'):
    with open(out_dir + '/global_step', 'r', encoding='utf8') as rf:
      dic_load = json.load(rf)
    GLOBAL_STEP = dic_load['global_step']
  else:
    GLOBAL_STEP = 0

  return GLOBAL_STEP


def get_filter_sizes(args):
  fsizes = args.filter_sizes
  args.filter_sizes = []
  for st in fsizes:
    args.filter_sizes.append(int(st))

def get_weight(train_iter):
  all_label = torch.zeros(4, )
  for batch in train_iter:
    label = batch.label
    label.data.sub_(1)
    for la in label.tolist():
      all_label[la] += 1

  weight = torch.sum(all_label) / all_label
  return weight


def get_cuda(args):
  USE_GPU = torch.cuda.is_available()
  cuda_str = 'cuda:' + args.CUDA_NUM
  DEVICE = torch.device(cuda_str if USE_GPU else "cpu")
  return USE_GPU, DEVICE


def get_logger(out_dir,):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  log_name = out_dir + '/log'
  fh = logging.FileHandler(log_name, mode='w')
  fh.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  return logger


def string_process(string):
  string = string.replace('\n', ' ')
  string = string.replace('\r', ' ')
  string = string.replace('\t', ' ')
  # string = string.replace(' ', '')
  string = string.strip('"')
  return string


def get_Model_Name(args):
  if args.model == 'cnn':
    model_name = args.model + '_fn' + str(args.filter_num) + '_fs' \
                 + args.filter_sizes + '_ed' + str(args.EMBEDDING_DIM) + '_dp' + str(args.dropout)
  else:
    model_name = args.model + '_hd' + str(args.HIDDEN_DIM) + \
                 '_ed' + str(args.EMBEDDING_DIM) + '_dp' + str(args.dropout)

  return model_name


def make_out_dir(args):
  model_name = get_Model_Name(args)
  dir_name = "task" + args.TASK_ID + '/' + model_name
  out_dir = os.path.abspath(os.path.join(os.path.curdir, dir_name))
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  return out_dir


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


def get_fields(taskid=None):
  text_field_train = data.Field(lower=True)
  text_field_dev = data.Field(lower=True)
  text_field_test = data.Field(lower=True)
  label_field = data.Field(sequential=False)

  dir_name = 'data/char/test/'
  if taskid:
    dir_name = 'data/char/ch_review' + taskid + '/'
  train, = data.TabularDataset.splits(path=dir_name, train='train.tsv', format='tsv',
                                     fields=[('text', text_field_train), ('label', label_field)])

  dev, = data.TabularDataset.splits(path=dir_name, train='dev.tsv', format='tsv',
                                     fields=[('text', text_field_dev), ('label', label_field)])

  test, = data.TabularDataset.splits(path='data/char/test/', train='test.tsv', format='tsv',
                                     fields=[('text', text_field_test)])

  text_field_train.build_vocab(train)
  text_field_dev.build_vocab(dev)
  text_field_test.build_vocab(test)
  label_field.build_vocab(train, dev)

  text_fields = {'train': text_field_train, 'dev': text_field_dev, 'test': text_field_test}
  dic_data = {'train': train, 'dev': dev, 'test': test}
  return dic_data, text_fields, label_field


def get_accuracy(truth, pred):
  assert len(truth) == len(pred)
  right = 0
  for i in range(len(truth)):
    if truth[i] == pred[i]:
      right += 1.0
  return right / len(truth)


def f1_score(truth, pred):
  TP = np.zeros(4,)
  FN = np.zeros(4,)
  FP = np.zeros(4,)
  precision = np.zeros(4,)
  recall = np.zeros(4,)

  for tr, pre in zip(truth, pred):
    if pre == tr:
      TP[tr] += 1
    else:
      FN[tr] += 1
      FP[pre] += 1

  for i in range(4):
    precision[i] = TP[i]/(TP[i] + FP[i])
    recall[i] = TP[i]/(TP[i] + FN[i])

  f1 = np.sum(2 * (precision * recall)/(precision + recall)) / 4
  return f1


def evaluate(model, data, loss_function):
  model.eval()
  avg_loss = 0.0
  truth_res = []
  pred_res = []
  for batch in data:
    sent, label = batch.text, batch.label
    label.data.sub_(1)
    truth_res += label.tolist()
    model.batch_size = len(label.data)
    pred = model(torch.transpose(sent, 0, 1))
    pred_label = pred.data.max(1)[1]
    pred_res += pred_label.tolist()
    loss = loss_function(pred, label)
    avg_loss += loss.data

  avg_loss /= len(data)
  acc = metrics.accuracy_score(truth_res, pred_res)
  # f1_1 = f1_score(truth_res, pred_res)
  f1 = metrics.f1_score(truth_res, pred_res, average='macro')

  return (avg_loss, acc, f1)


def train_epoch_progress(model, train_iter, loss_function, optimizer):
  model.train()
  avg_loss = 0.0
  truth_res = []
  pred_res = []
  count = 0
  for batch in train_iter:
    sent, label = batch.text, batch.label
    label.data.sub_(1)
    truth_res += label.tolist()
    pred = model(torch.transpose(sent, 0, 1))
    pred_label = pred.data.max(1)[1]
    pred_res += [x for x in pred_label]
    model.zero_grad()
    loss = loss_function(pred, label)
    avg_loss += loss.data
    count += 1
    loss.backward()
    optimizer.step()

  avg_loss /= len(train_iter)
  acc = metrics.accuracy_score(truth_res, pred_res)
  f1 = metrics.f1_score(truth_res, pred_res, average='macro')
  return avg_loss, acc, f1


def show_info(logger, writer, epoch, train_info, dev_info):
  logger.info('[EPOCH{}] Train: loss {}, acc {}, f1 {}'.format(epoch, train_info[0], train_info[1], train_info[2]))
  logger.info('[EPOCH{}] Dev: loss {}, acc {}, f1 {}\n'.format(epoch, dev_info[0], dev_info[1], dev_info[2]))

  writer.add_scalars('Loss', {'train': train_info[0], 'dev': dev_info[0]}, epoch)
  writer.add_scalars('Accuracy', {'train': train_info[1], 'dev': dev_info[1]}, epoch)
  writer.add_scalars('F1', {'train': train_info[2], 'dev': dev_info[2]}, epoch)


def train_start(EPOCHS, model, train_iter, dev_iter, loss_function,
                optimizer, writer, logger, out_dir, global_step):
  dev_info = evaluate(model, dev_iter, loss_function)
  train_info = evaluate(model, train_iter, loss_function)
  show_info(logger, writer, 0, train_info, dev_info)
  best_dev_f1 = dev_info[2]
  for epoch in range(EPOCHS):
      train_info = train_epoch_progress(model, train_iter, loss_function, optimizer)
      dev_info = evaluate(model, dev_iter, loss_function)
      show_info(logger, writer, epoch+1+global_step, train_info, dev_info)

      with open(out_dir+'/global_step', 'w', encoding='utf8') as wf:
        dic_glob = {'global_step': epoch+1+global_step}
        json.dump(dic_glob, wf)

      if dev_info[2] > best_dev_f1:
          if os.path.exists(out_dir + '/best_model.pth'):
              os.system('rm ' + out_dir + '/best_model.pth')
          best_dev_f1 = dev_info[2]
          torch.save(model.state_dict(), out_dir + '/best_model' + '.pth')
