import logging
import argparse

# Imports from the module we wrapped our functions
from audiolib import gtzan_parser
from audiolib.struct import RawAudio, to_melspectrogram
from audiolib.ttsplit import ttsplit_cml, ttsplit_cnn

# List with all the genres available on the GTZAN dataset
classes = [
  'metal', 'disco', 'classical', 'hiphop', 'jazz',
  'country', 'pop', 'blues', 'reggae', 'rock'
]

def main(args):
  # Validate how to retrieve the data
  if args.fread == 'RAW':
    audio = RawAudio('RAW', file_path = args.directory, classes = classes)
  elif args.fread == 'NPY':
    audio = RawAudio('NPY', songs = args.songs, genres = args.genres)
  else:
    raise Exception('fread invalid')

  # Get the data as numpy arrays
  songs, genres = audio.get_data()

  if args.save and args.fread == 'RAW':
    RawAudio.save_data(songs = songs, genres = genres, file_path = args.savedir)

  if args.ctype == 'CML':
    # Get audio features from songs
    ds = get_features(songs)
  elif args.ctype == '1D' or '2D':
    # Convert the songs to melspectrograms
    melspecs = to_melspectrogram(songs)
  else:
    raise Exception('ctype invalid')

  # training exec times to ensure was no split luck
  for it in range(args.exec):
    if args.ctype == 'CML':
      # split the dataset in train and test
      x_train, y_train, x_test, y_test, x_val, y_val = ttsplit_cml(ds)
      model = train_cml(x_train, y_train, x_val, y_val)
    else:
      # split the dataset in train and test for cnn models
      x_train, y_train, x_test, y_test, x_val, y_val = ttsplit_cnn(melspecs, args.ctype)
      model = train_cnn(x_train, y_train, x_val, y_val)

  # Deallocate memory
  del songs
  del melspecs
  del genres

def train_cnn(x_train, y_train, x_val, y_val):
  pass

def train_cml(x_train, y_train, x_val, y_val):
  pass

if __name__ == '__main__':
  parser = gtzan_parser()
  args = parser.parse_args()
  main(args)