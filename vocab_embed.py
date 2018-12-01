import argparse
import random
import time
import numpy as np
from utils import get_fields
from collections import defaultdict


def filter_embed_txt(embed_file, text_fields, new_embed):
  wf = open(new_embed, 'w', encoding='utf8')
  rf = open(embed_file, 'r', encoding='utf8')
  rf.readline()
  t = 0
  for line in rf:
    t += 1
    if t % 10000 == 0:
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


def load_embeddings(embedding_dim, text_fields, logger=None, emb_file=None):
  def _return_unk():
    return 0

  if not emb_file:
    if logger:
      logger.info('Randomly initialize word embeddings')
    else:
      print('Randomly initialize word embeddings')
    vocab_list = []
    for tr in text_fields['train'].vocab.itos:
      vocab_list.append(tr)
    for de in text_fields['dev'].vocab.itos:
      vocab_list.append(de)
    for te in text_fields['test'].vocab.itos:
      vocab_list.append(te)
    vocab_dic = {w: i for i, w in enumerate(vocab_list)}
    # pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_list), embedding_dim))
    # pretrained_embeddings[0] = 0
    pretrained_embeddings = None
    return pretrained_embeddings, vocab_list, vocab_dic
  else:
    vocab_dic = defaultdict(_return_unk)
    emb_dic, _ = load_embed_txt(emb_file)
    pretrained_embeddings = []
    vocab_list = ['<unk>', '<pad>', '#sep#']
    vocab_dic['<unk>'] = 0
    vocab_dic['<pad>'] = 1
    vocab_dic['#sep#'] = 2
    pretrained_embeddings.append([random.random() for _ in range(embedding_dim)])
    pretrained_embeddings.append([random.random() for _ in range(embedding_dim)])
    pretrained_embeddings.append([random.random() for _ in range(embedding_dim)])
    count1 = 0
    wf = open('un_token', 'w', encoding='utf8')
    for word in text_fields['train'].vocab.itos:
      if word in emb_dic:
        vocab_list.append(word)
        vocab_dic[word] = len(vocab_dic)
        pretrained_embeddings.append(emb_dic[word])
      else:
        wf.write(word + '\n')
        count1 += 1
        # pretrained_embeddings.append([random.random() for _ in range(embedding_dim)])

    count2 = 0
    for word in text_fields['dev'].vocab.itos:
      if word not in emb_dic and word not in vocab_dic:
        count2 += 1
        wf.write(word + '\n')
      if word in emb_dic and word not in vocab_dic:
        vocab_list.append(word)
        vocab_dic[word] = len(vocab_dic)
        pretrained_embeddings.append(emb_dic[word])

    count3 = 0
    for word in text_fields['test'].vocab.itos:
      if word not in emb_dic and word not in vocab_dic:
        count3 += 1
        wf.write(word + '\n')
      if word in emb_dic and word not in vocab_dic:
        vocab_list.append(word)
        vocab_dic[word] = len(vocab_dic)
        pretrained_embeddings.append(emb_dic[word])

    if logger:
      logger.info('Train vocabulary size: {}, {} of them are not in {}'.format(len(text_fields['train'].vocab.itos), count1, emb_file))
      logger.info('Dev vocabulary size: {}, unknown tokens: {}'.format(len(text_fields['dev'].vocab.itos), count2))
      logger.info('Test vocabulary size: {}, unknown tokens: {}'.format(len(text_fields['test'].vocab.itos), count3))
    else:
      print('Train vocabulary size: {}, {} of them are not in {}'.format(len(text_fields['train'].vocab.itos), count1, emb_file))
      print('Dev vocabulary size: {}, unknown tokens: {}'.format(len(text_fields['dev'].vocab.itos), count2))
      print('Test vocabulary size: {}, unknown tokens: {}'.format(len(text_fields['test'].vocab.itos), count3))

    wf.close()
    return pretrained_embeddings, vocab_list, vocab_dic


if __name__ == '__main__':
  args = argparse.ArgumentParser()
  args.add_argument('--split', dest='split', type=str)
  args = args.parse_args()
  split = args.split
  dic_data, text_fields, label_field = get_fields('data/'+split+'/',)
  embed_file = 'embed3'
  new_embed_file = split + '_' + embed_file
  filter_embed_txt(embed_file, text_fields, new_embed_file)
  load_embeddings(200, text_fields, emb_file=new_embed_file)
