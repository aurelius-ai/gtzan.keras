import numpy as np

class MusicGenerator(object):
  """Keras Generator for Music data (Using MIXUP)"""
  def __init__(self, arg):
    super(MusicGenerator, self).__init__()
    self.arg = arg

def splitsongs_melspect(self, X, y, cnn_type = '1D'):
  temp_X = []
  temp_y = []

  for i, song in enumerate(X):
    song_slipted = np.split(song, self.augment_factor)
    for s in song_slipted:
      temp_X.append(s)
      temp_y.append(y[i])

  temp_X = np.array(temp_X)
  temp_y = np.array(temp_y)

  if not cnn_type == '1D':
    temp_X = temp_X[:, np.newaxis]
    
  return temp_X, temp_y 