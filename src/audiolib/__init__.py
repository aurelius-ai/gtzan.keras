from .features import *
from .utils import *
from .models import *
from .struct import *
from .ttsplit import *

import argparse

def gtzan_parser():
  parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN')

  # Required arguments
  parser.add_argument('--ctype', 
      help='Choose the type of the Classifier (CML, 1D or 2D)',
      type=str, required=True)
  parser.add_argument('--fread',
      help='Choose How to read the files (RAW or NPY)', 
      type=str, required=True)

  # Almost optional arguments. Should be filled according to the option of the requireds
  parser.add_argument('-d', '--directory', 
      help='Path to the root directory with GTZAN files', type=str)
  parser.add_argument('-s', '--save', 
      help='Save the RAW audios to NPY format', type=bool)
  parser.add_argument('--songs', 
      help='File path to the NPY file with songs as numpy arrays', type=str)
  parser.add_argument('--genres', 
      help='File path to the NPY file with genres one-hot encoded as numpy arrays', type=str)
  parser.add_argument('--savedir', 
      help='Path to where save the NPY files', type=str)
  parser.add_argument('--model', 
      help='Choose the 2D CNN pre-trained model/architecture', type=str)
  parser.add_argument('--exec', 
      help='Number of times to execute the training process', nargs='?', const=5, type=int)

  return parser