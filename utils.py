import torch
import numpy as np
from torchtext import data
import os
import time
import random
from collections import defaultdict
import sys



def filter_embed_txt(embed_file, text_fields):
  wf = open('new_embed', 'w', encoding='utf8')
  rf = open(embed_file, 'r', encoding='utf8')
  rf.readline()
  t = 0
  for line in rf:
    t += 1
    if t % 1000 == 0:
      print(t)
    tokens = line.strip().split(" ")
    word = tokens[0]
    if (word in text_fields['train'].vocab.stoi
            or word in text_fields['dev'].vocab.stoi
            or word in text_fields['test'].vocab.stoi):
      wf.write(line)
  wf.close()
  rf.close()


def load_embed_txt(embed_file):
  emb_dict = dict()
  emb_size = None
  with open(embed_file, 'r', encoding='utf8') as f:
    f.readline()
    for line in f:
      tokens = line.strip().split(" ")
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        assert emb_size == len(vec), "all embedding size should be the same"
      else:
        emb_size = len(vec)

  return emb_dict, emb_size


def load_embeddings(args, text_fields, emb_file=None):
  def _return_unk():
    return text_fields['train'].vocab.stoi['<unk>']

  vocab_dic = defaultdict(_return_unk)

  if emb_file:
    print('load pre-trained word embeddings')
    start_time = time.time()
    emb_dic, _ = load_embed_txt(emb_file)
    print('complete loading embeddings in {} seconds'.format(time.time() - start_time))
    pretrained_embeddings = []
    vocab_list = []
    count = 0
    for word in text_fields['train'].vocab.itos:
      vocab_list.append(word)
      vocab_dic[word] = len(vocab_dic)
      if word in emb_dic:
        pretrained_embeddings.append(emb_dic[word])
      else:
        count += 1
        pretrained_embeddings.append([random.random() for _ in range(args.EMBEDDING_DIM)])

    print('{}/{} train vocabulary tokens'
          ' are not in pre-trained word embeddings'.format(count, len(text_fields['train'].vocab.itos)))

    count = 0
    for word in text_fields['dev'].vocab.itos:
      if word not in emb_dic and word not in vocab_dic:
        count += 1
      if word in emb_dic and word not in vocab_dic:
        vocab_list.append(word)
        vocab_dic[word] = len(vocab_dic)
        pretrained_embeddings.append(emb_dic[word])

    print('{}/{} dev vocabulary tokens'
          ' are unknown tokens'.format(count, len(text_fields['dev'].vocab.itos)))

    count = 0
    for word in text_fields['test'].vocab.itos:
      if word not in emb_dic and word not in vocab_dic:
        count += 1
      if word in emb_dic and word not in vocab_dic:
        vocab_list.append(word)
        vocab_dic[word] = len(vocab_dic)
        pretrained_embeddings.append(emb_dic[word])

    print('{}/{} test vocabulary tokens'
          ' are unknow tokens'.format(count, len(text_fields['test'].vocab.itos)))

    return pretrained_embeddings, vocab_list, vocab_dic


def string_process(string):
  string = string.replace('\n', ' ')
  string = string.replace('\r', ' ')
  string = string.replace('\t', ' ')
  # string = string.replace(' ', '')
  string = string.strip('"')
  return string


def get_Model_Name(args):
  model_name = args.model + '_bs' + str(args.BATCH_SIZE) + '_hd' + str(args.HIDDEN_DIM) +\
               '_ed' + str(args.EMBEDDING_DIM)

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

  dir_name = 'data/test/'
  if taskid:
    dir_name = 'data/ch_review' + taskid + '/'
  train, = data.TabularDataset.splits(path=dir_name, train='train.tsv', format='tsv',
                                     fields=[('text', text_field_train), ('label', label_field)])

  dev, = data.TabularDataset.splits(path=dir_name, train='dev.tsv', format='tsv',
                                     fields=[('text', text_field_dev), ('label', label_field)])

  test, = data.TabularDataset.splits(path='data/test/', train='test.tsv', format='tsv',
                                     fields=[('text', text_field_test)])

  text_field_train.build_vocab(train)
  text_field_dev.build_vocab(dev)
  text_field_test.build_vocab(test)
  label_field.build_vocab(train, dev)

  text_fields = {'train': text_field_train, 'dev': text_field_dev, 'test': text_field_test}
  dic_data = {'train': train, 'dev': dev, 'test': test}
  return dic_data, text_fields, label_field


def load_review_test(text_field, label_field, batch_size, DEVICE):
  train, dev, test = data.TabularDataset.splits(path='data/test/', train='train.tsv',
                                                validation='dev.tsv', test='dev.tsv', format='tsv',
                                                fields=[('text', text_field), ('label', label_field)])

  text_field.build_vocab(train, dev, test)
  label_field.build_vocab(train, dev, test)
  idx2word = text_field.vocab.itos

  text_field2 = data.Field(lower=True)
  label_field2 = data.Field(sequential=False)

  train, dev, test = data.TabularDataset.splits(path='data/test/', train='test.tsv',
                                                validation='test.tsv', test='test.tsv', format='tsv',
                                                fields=[('text', text_field2), ('label', label_field2)])

  text_field2.build_vocab(train, dev, test)
  label_field2.build_vocab(train, dev, test)
  idx2word2 = text_field2.vocab.itos

  tt = 0
  for word in idx2word2:
    if word not in idx2word:
      tt += 1

  train_iter, dev_iter, test_iter = \
    data.BucketIterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size),
                               sort_key=lambda x: len(x.text), repeat=False, device=DEVICE)

  dic_data = {'train': train_iter, 'dev': dev_iter, 'test': test_iter}
  return dic_data


def get_accuracy(truth, pred):
  assert len(truth) == len(pred)
  right = 0
  for i in range(len(truth)):
    if truth[i] == pred[i]:
      right += 1.0
  return right / len(truth)


def evaluate(model, data, loss_function, name, log_file):
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
  acc = get_accuracy(truth_res, pred_res)
  log_file.write(name + ': loss %.2f acc %.1f\t' % (avg_loss, acc * 100))
  return acc


def train_epoch_progress(model, train_iter, loss_function, optimizer):
  model.train()
  avg_loss = 0.0
  truth_res = []
  pred_res = []
  count = 0
  for batch in train_iter:
    sent, label = batch.text, batch.label
    label.data.sub_(1)
    truth_res += list(label.data)
    model.batch_size = len(label.data)
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
  acc = get_accuracy(truth_res, pred_res)
  return avg_loss, acc


def train_start(EPOCHS, model, dic_iter, loss_function, optimizer, log_file, out_dir):
  best_dev_acc = evaluate(model, dic_iter['dev'], loss_function, 'Initial test', log_file)
  for epoch in range(EPOCHS):
      avg_loss, acc = train_epoch_progress(model, dic_iter['train'], loss_function, optimizer)
      log_file.write('\n[EPOCH%d] Train: loss %.2f acc %.1f\t' % (epoch, avg_loss, acc*100))
      dev_acc = evaluate(model, dic_iter['dev'], loss_function, 'Dev', log_file)
      if dev_acc > best_dev_acc:
          if os.path.exists(out_dir + '/best_model.pth'):
              os.system('rm ' + out_dir + '/best_model.pth')
          best_dev_acc = dev_acc
          torch.save(model.state_dict(), out_dir + '/best_model' + '.pth')


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