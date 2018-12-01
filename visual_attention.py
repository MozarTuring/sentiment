import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import *

# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
myfont = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')

def showAttention(input_sentence, output_words, attentions, i):
  # Set up figure with colorbar
  # print(attentions)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.numpy(), cmap='bone')
  fig.colorbar(cax)

  # Set up axes
  ax.set_xticklabels([''] + input_sentence +
                     ['<EOS>'], rotation=90, fontdict={'fontproperties': myfont})
  ax.set_yticklabels([''] + output_words)

  # Show label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.savefig('figures/fig' + str(i))
