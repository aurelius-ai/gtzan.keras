import os
import librosa
import numpy as np

# @Function: to_melspectrogram
# @Description: Convert a set of audio songs (np arrays) to melspectrograms
def to_melspectrogram(songs, n_fft = 2048, hop_length = 512):
  # Transformation function
  melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
    hop_length = hop_length)

  # map transformation of input songs to melspectrogram using log-scale
  tsongs = map(melspec, songs)
  return np.array(list(tsongs))