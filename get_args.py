import argparse


args = argparse.ArgumentParser()
args.add_argument('--cu', dest='CUDA_NUM', type=str)
args.add_argument('--m', dest='model', help='specify the mode to use (default: lstm)')
args.add_argument('--dropout', dest='dropout', type=float)
args.add_argument('--emdim', dest='EMBEDDING_DIM', type=int)
args.add_argument('--bsize', dest='BATCH_SIZE', type=int)
args.add_argument('--p', dest='paral', default=False, type=bool)


def get_args_train():
  global args
  args.add_argument('--id', dest='TASK_ID', type=str)
  args.add_argument('--epos', dest='EPOCHS', type=int)
  args.add_argument('--run', dest='run', default='', type=str)


def get_args_cnn(mode=None):
  global args
  args.add_argument('--fnum', dest='filter_num', type=int)
  args.add_argument('--fsizes', dest='filter_sizes', type=str)
  if mode == 'train':
    get_args_train()
  args = args.parse_args()
  return args


def get_args_rnn(mode=None):
  global args
  args.add_argument('--attn', dest='attn_method', help='general, dot, concat')
  args.add_argument('--hdim', dest='HIDDEN_DIM', type=int)
  if mode == 'train':
    get_args_train()
  args = args.parse_args()
  return args
