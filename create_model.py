import models
import models2
import models3
import models4
import torch.nn as nn


def create_model(args, vocab_ls, label_field, USE_GPU, DEVICE, NUM_ELE, logger=None):
  if args.model == 'lstm':
    model = models.LSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                 vocab_size=len(vocab_ls), label_size=len(label_field.vocab.itos) - 1,
                                 use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE)

  if args.model == 'lstm_attention':
    model = models2.LSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                  vocab_size=len(vocab_ls.vocab), label_size=len(label_field.vocab.itos) - 1,
                                  use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE)

  if args.model == 'bilstm':
    model = models.BiLSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                   vocab_size=len(vocab_ls), label_size=len(label_field.vocab.itos) - 1,
                                   use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE)

  if args.model == 'bilstm_attention':
    model = models4.BiLSTMSentiment(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                                    vocab_size=len(vocab_ls), label_size=len(label_field.vocab.itos) - 2,
                                    use_gpu=USE_GPU, batch_size=args.BATCH_SIZE, devic=DEVICE,
                                    attn_method=args.attn_method, num_ele=NUM_ELE, dropout=args.dropout)

  if args.model == 'gru':
    embedding = nn.Embedding(len(vocab_ls), args.EMBEDDING_DIM)
    model = models.GRUSentiment(hidden_size=args.HIDDEN_DIM, embedding=embedding, n_layers=1, dropout=0)

  if args.model == 'gru_attention':
    embedding = nn.Embedding(len(vocab_ls), args.EMBEDDING_DIM)
    model = models2.GRUSentiment(hidden_size=args.HIDDEN_DIM, embedding=embedding, n_layers=1, dropout=0)

  if args.model == 'cnn':
    model = models3.CNNSentiment(vocab_size=len(vocab_ls), embedding_dim=args.EMBEDDING_DIM, n_filters=args.filter_num,
                                filter_sizes=args.filter_sizes, output_dim=len(label_field.vocab.itos)-2,
                                dropout=args.dropout)

  if logger:
    logger.info('Creating {} model'.format(args.model))
  else:
    print('Creating {} model'.format(args.model))

  if args.paral:
    model = nn.DataParallel(model)

  model.to(DEVICE)

  return model