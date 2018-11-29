import torch
import json
import numpy as np
from torchtext import data
import os
from sklearn import metrics
import logging
from tensorboardX import SummaryWriter
import re
import sys
from zhon.hanzi import punctuation


def change_vocab(field, dic, ls):
  field.vocab.itos = ls
  field.vocab.stoi = dic
  field.vocab.itos = ls
  field.vocab.stoi = dic


def load_model(out_dir, model, logger):
  if os.path.exists(out_dir + '/best_model.pth'):
    model.load_state_dict(torch.load(out_dir + '/best_model.pth'))
    logger.info('Initial model is loaded')
  else:
    logger.info('Initial model is fresh')

  return model


def get_writer(args, out_dir):
  if args.run:
    writer = SummaryWriter(log_dir='runs/run' + args.run)
  else:
    writer = SummaryWriter(log_dir=out_dir)
  return writer


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

  _, indx = torch.sort(all_label)
  balance = torch.zeros(4, )
  balance[indx[0]] = 1.5
  balance[indx[1]] = 1
  balance[indx[2]] = 1
  balance[indx[3]] = 0.9

  weight = torch.sum(all_label) / all_label
  weight *= balance
  return weight


def get_weights(train_iter, label_field):
  SHAPE_COARSE = (6, 3)
  SHAPE_FINE = (20, 4)
  all_label = torch.zeros(SHAPE_FINE)
  all_coarse = torch.zeros(SHAPE_COARSE)
  for i, batch in enumerate(train_iter):
    label = batch.label
    label.data.sub_(2)
    coarse_label = map2coarse(label.transpose(0, 1).tolist(), label_field)
    for j, lab in enumerate(label.tolist()):
      for la in lab:
        all_label[j, la] += 1
    for j, coa in enumerate(torch.tensor(coarse_label).reshape(-1, 6).transpose(0, 1).tolist()):
      for co in coa:
        all_coarse[j, co] += 1

  all_label += 1
  weights = torch.sum(all_label, dim=-1, keepdim=True) / all_label
  balance = torch.ones(1, SHAPE_FINE[1])
  balance[0, 0] = 0.9
  weights *= balance

  all_coarse += 1
  w_coarse = torch.sum(all_coarse, dim=-1, keepdim=True) / all_coarse
  b_coarse = torch.ones(1, SHAPE_COARSE[1])
  w_coarse *= b_coarse
  return (weights, w_coarse)


def get_cuda(args):
  USE_GPU = torch.cuda.is_available()
  cuda_str = 'cuda:' + args.CUDA_NUM
  DEVICE = torch.device(cuda_str if USE_GPU else "cpu")
  return USE_GPU, DEVICE


def get_logger(out_dir,):
  logger = logging.getLogger(out_dir)
  logger.setLevel(logging.INFO)
  log_name = out_dir + '/log'
  fh = logging.FileHandler(log_name, mode='w')
  fh.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s")
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  return logger


def string_process(string):
  # string = string.replace(',', '<sep>')
  # string = string.replace('，', '<sep>')
  # string = string.replace('。', '<sep>')
  # string = string.replace(' ', '<sep>')
  string = string.strip('"')
  string = re.sub('\d', '0', string)
  string = re.sub(r"[{}]+".format(punctuation), '<sep>', string)
  string = string.replace('\n', '<sep>')
  string = string.replace('\r', '<sep>')
  string = string.replace('\t', '<sep>')
  return string


def get_Model_Name(args):
  if args.model == 'cnn':
    model_name = args.model + '_fn' + str(args.filter_num) + '_fs' \
                 + args.filter_sizes + '_ed' + str(args.EMBEDDING_DIM) + '_dp' + str(args.dropout)
  else:
    model_name = args.model + '_hd' + str(args.HIDDEN_DIM) + \
                 '_ed' + str(args.EMBEDDING_DIM) + '_dp' + str(args.dropout)

  return model_name


def make_out_dir(args, NUM_ELE):
  model_name = get_Model_Name(args)
  if args.TASK_ID:
    dir_name = "task" + args.TASK_ID + '/' + model_name + '/running'
  elif NUM_ELE == 6:
    dir_name = 'coarse/' + model_name + '/running'
  elif NUM_ELE == 20:
    dir_name = 'task/' + model_name + '/running'
  elif NUM_ELE == 26:
    dir_name = 'together/' + model_name + '/running'
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


def get_fields(path, taskid=None, mode=None):
  text_field_train = data.Field(lower=True)
  text_field_dev = data.Field(lower=True)
  text_field_test = data.Field(lower=True)
  label_field = data.Field()

  if taskid == None:
    if mode == 'test':
      dir_name = path + 'testb/'
    else:
      dir_name = path + 'ch_review/'
  else:
    dir_name = path + 'ch_review' + taskid + '/'

  train, = data.TabularDataset.splits(path=dir_name, train='train.tsv', format='tsv',
                                     fields=[('text', text_field_train), ('label', label_field)])

  dev, = data.TabularDataset.splits(path=dir_name, train='dev.tsv', format='tsv',
                                     fields=[('text', text_field_dev), ('label', label_field)])

  test, = data.TabularDataset.splits(path=path+'testb/', train='test.tsv', format='tsv',
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


def sgn(x, thl=-5, thu=-3):
  if x > thu:
    return 2
  elif x < thl:
    return 0
  else:
    return 1


def map2coarse(fine_ls, label_field):
  lls = []
  for lis in fine_ls:
    lss = list(map(lambda x: int(label_field.vocab.itos[x+2]), lis))
    lls += list(map(lambda x: sgn(x),
                    [sum(lss[:3]), sum(lss[3:7]), sum(lss[7:10]),
                     sum(lss[10:14]), sum(lss[14:18]), sum(lss[18:])]))
  return lls


def epoch_progress(model, train_iter, loss_functions, label_field, DEVICE, optimizer=None):
  if optimizer:
    model.train()
  else:
    model.eval()
  avg_loss = 0.0
  truth_res = []
  pred_ls = []
  coarse_ls = []
  coarse_truth = []
  count = 0
  batch_size = None
  for batch in train_iter:
    sent, label = batch.text, batch.label
    if not batch_size:
      batch_size = label.shape[1]

    label.data.sub_(2)
    coarse_label_ls = map2coarse(label.transpose(0, 1).tolist(), label_field) # batch_size * 6 to list
    coarse_truth += coarse_label_ls
    coarse_label = torch.tensor(coarse_label_ls, device=DEVICE).reshape(-1, 6).transpose(0, 1)
    truth_res += label.reshape(-1, ).tolist()
    pred = model(torch.transpose(sent, 0, 1))
    pred_label = [pred[0].data.max(-1)[1], pred[1].data.max(-1)[1]]
    pred_ls += pred_label[0].reshape(-1, ).tolist() # pred_label[0].shape is 20 * batch_size
    coarse_ls += pred_label[1].transpose(0, 1).reshape(-1, ).tolist() # pred_label[1].shape is 6 * batch_size
    model.zero_grad()
    loss = 0
    for i in range(20):
      loss += loss_functions[i](pred[0][i], label[i])
    for i in range(6):
      loss += 2*loss_functions[20+i](pred[1][i], coarse_label[i])
    avg_loss += loss.data
    count += 1
    if optimizer:
      loss.backward()
      optimizer.step()

  avg_loss /= len(train_iter)
  acc = metrics.accuracy_score(truth_res, pred_ls)
  f1 = metrics.f1_score(truth_res, pred_ls, average='macro')
  cf_matrix = metrics.confusion_matrix(truth_res, pred_ls)

  coarse_acc = metrics.accuracy_score(coarse_truth, coarse_ls)
  coarse_f1 = metrics.f1_score(coarse_truth, coarse_ls, average='macro')
  coarse_cf = metrics.confusion_matrix(coarse_truth, coarse_ls)

  return [(avg_loss, acc, f1, cf_matrix), (0, coarse_acc, coarse_f1, coarse_cf)]


def summ(arr):
  return np.sum(arr, axis=1, keepdims=False)


def my_f1(arr):
  TP = np.zeros(4,)
  FP = np.zeros(4,)
  FN = np.zeros(4,)
  for i in range(4):
    TP[i] = arr[i,i]
    FP[i] = np.sum(arr[:,i]) - TP[i]
    FN[i] = np.sum(arr[i,:]) - TP[i]
  pre = TP / (TP + FP)
  rec = TP / (TP + FN)
  return np.sum(2 * pre * rec / (pre + rec)) / 4


def show_info(logger, writer, epoch, train_info, dev_info, coarse=False):
  logger.info('[EPOCH{}] Train: loss {}, acc {}, f1 {}, confusion matrix\n {}, {}'.
              format(epoch, train_info[0], train_info[1], train_info[2], train_info[3], summ(train_info[3])))
  logger.info('[EPOCH{}] Dev: loss {}, acc {}, f1 {}, confusion matrix\n {}, {}\n'.
              format(epoch, dev_info[0], dev_info[1], dev_info[2], dev_info[3], summ(dev_info[3])))

  if not coarse:
    writer.add_scalars('Loss', {'train': train_info[0], 'dev': dev_info[0]}, epoch)
    writer.add_scalars('Accuracy', {'train': train_info[1], 'dev': dev_info[1]}, epoch)
    writer.add_scalars('F1', {'train': train_info[2], 'dev': dev_info[2]}, epoch)


def train_start(EPOCHS, model, train_iter, dev_iter, loss_functions,
                optimizer, writer, logger, out_dir, label_field, DEVICE, global_step=0):
  train_info = epoch_progress(model, train_iter, loss_functions, label_field, DEVICE,)
  dev_info = epoch_progress(model, dev_iter, loss_functions, label_field, DEVICE,)
  show_info(logger, writer, global_step, train_info[0], dev_info[0])
  show_info(logger, writer, global_step, train_info[1], dev_info[1], True)
  best_dev_f1 = dev_info[0][2]
  logger.info('Initial best dev f1 is {}\n'.format(best_dev_f1))
  for epoch in range(EPOCHS):
      train_info = epoch_progress(model, train_iter, loss_functions, label_field, DEVICE, optimizer)
      dev_info = epoch_progress(model, dev_iter, loss_functions, label_field, DEVICE,)
      show_info(logger, writer, epoch+1+global_step, train_info[0], dev_info[0])
      show_info(logger, writer, epoch+1+global_step, train_info[1], dev_info[1], True)

      if dev_info[0][2] > best_dev_f1:
          if os.path.exists(out_dir + '/best_model.pth'):
              os.system('rm ' + out_dir + '/best_model.pth')
          logger.info('best dev f1 is improved from {} to {}\n'.format(best_dev_f1, dev_info[0][2]))
          best_dev_f1 = dev_info[0][2]
          torch.save(model.state_dict(), out_dir + '/best_model' + '.pth')

      with open(out_dir+'/global_step', 'w', encoding='utf8') as wf:
        dic_glob = {'global_step': epoch+1+global_step}
        json.dump(dic_glob, wf)


# def evaluate(model, data, loss_functions, label_field):
#   model.eval()
#   avg_loss = 0.0
#   truth_res = []
#   pred_res = []
#   coarse_pred = []
#   coarse_truth = []
#   for batch in data:
#     sent, label = batch.text, batch.label
#     label.data.sub_(2)
#     coarse_truth += map2coarse(label.transpose(0, 1).tolist(), label_field)
#     truth_res += label.reshape(-1,).tolist()
#     model.batch_size = len(label.data)
#     pred = model(torch.transpose(sent, 0, 1))
#     pred_label = pred.data.max(-1)[1]   # NUM_ELE * BATCH_SIZE
#     # coarse_pred += map2coarse(pred_label.transpose(0, 1).tolist(), label_field)
#     pred_res += pred_label.reshape(-1,).tolist()
#     loss = loss_functions[0](pred[0], label[0])
#     for i in range(len(loss_functions)-1):
#       loss += loss_functions[i+1](pred[i+1], label[i+1])
#     avg_loss += loss.data
#
#   avg_loss /= len(data)
#   acc = metrics.accuracy_score(truth_res, pred_res)
#   f1 = metrics.f1_score(truth_res, pred_res, average='macro')
#   cf_matrix = metrics.confusion_matrix(truth_res, pred_res)
#
#   coarse_acc = metrics.accuracy_score(coarse_truth, coarse_pred)
#   coarse_f1 = metrics.f1_score(coarse_truth, coarse_pred, average='macro')
#   coarse_cf = metrics.confusion_matrix(coarse_truth, coarse_pred)
#
#   return [(avg_loss, acc, f1, cf_matrix), (None, coarse_acc, coarse_f1, coarse_cf)]
