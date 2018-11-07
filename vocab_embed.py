import random
import time
from collections import defaultdict


def filter_embed_txt(embed_file, text_fields, new_embed):
  wf = open(new_embed, 'w', encoding='utf8')
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


def load_embeddings(args, text_fields, logger, emb_file=None):
  def _return_unk():
    return text_fields['train'].vocab.stoi['<unk>']

  vocab_dic = defaultdict(_return_unk)

  if emb_file:
    logger.info('load pre-trained word embeddings')
    start_time = time.time()
    emb_dic, _ = load_embed_txt(emb_file)
    logger.info('complete loading embeddings in {} seconds'.format(time.time() - start_time))
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

    logger.info('Train vocabulary size: {}, unknown tokens: {}'.format(len(text_fields['train'].vocab.itos), count))

    count = 0
    for word in text_fields['dev'].vocab.itos:
      if word not in emb_dic and word not in vocab_dic:
        count += 1
      if word in emb_dic and word not in vocab_dic:
        vocab_list.append(word)
        vocab_dic[word] = len(vocab_dic)
        pretrained_embeddings.append(emb_dic[word])

    logger.info('Dev vocabulary size: {}, unknown tokens: {}'.format(len(text_fields['dev'].vocab.itos), count))

    count = 0
    for word in text_fields['test'].vocab.itos:
      if word not in emb_dic and word not in vocab_dic:
        count += 1
      if word in emb_dic and word not in vocab_dic:
        vocab_list.append(word)
        vocab_dic[word] = len(vocab_dic)
        pretrained_embeddings.append(emb_dic[word])

    logger.info('Test vocabulary size: {}, unknown tokens: {}'.format(len(text_fields['test'].vocab.itos), count))

    return pretrained_embeddings, vocab_list, vocab_dic