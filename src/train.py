import os
import gc
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras import backend as K

# Imports from the module we wrapped our functions
from audiolib import gtzan_parser
from audiolib.models import get_cnn1d
from audiolib.utils import splitsongs
from audiolib.utils import RawAudio, to_melspectrogram
from audiolib.utils import ttsplit_cml, ttsplit_cnn

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# List with all the genres available on the GTZAN dataset
classes = [
  'metal', 'disco', 'classical', 'hiphop', 'jazz',
  'country', 'pop', 'blues', 'reggae', 'rock'
]

# python train.py --ctype 1D --fread NPY --songs ../dataset/GTZAN/songs.npy --genres ../dataset/GTZAN/genres.npy --exec 5
# python train.py --ctype 1D --fread RAW -d ../dataset/GTZAN/ --savedir ../dataset/GTZAN/ --exec 5

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
  print("Original sound array shape: {}".format(songs.shape))

  # Save the data asked for
  if args.save and args.fread == 'RAW':
    RawAudio.save_data(songs = songs, genres = genres, file_path = args.savedir)

  # Split the songs array in multiple pieces of songs
  songs, genres = splitsongs(songs, genres)
  print("After split sound array shape: {}".format(songs.shape))
 
  # Choose how to process the songs according to classifier type
  if args.ctype == 'CML':
    # Get audio features from songs
    ds = get_features(songs)
  elif args.ctype == '1D' or '2D':
    # Convert the songs to melspectrograms
    melspecs = to_melspectrogram(songs)
  else:
    raise Exception('ctype invalid')

  # Deallocate memory
  del songs

  # training exec times to ensure was no split luck
  for it in range(args.exec):
    print("Execution %d" % it)
    
    # Variable with training history for keras
    hist = None

    # Choosing training type
    if args.ctype == 'CML':
      # split the dataset in train and test
      x_train, y_train, x_test, y_test, x_val, y_val = ttsplit_cml(ds, genres)
      print("train, test and val size: {}, {}, {}".format(x_train.shape, x_test.shape, x_val.shape))
      
      # Start training process
      model = train_cml(x_train, y_train, x_val, y_val)
    else:
      # split the dataset in train and test for cnn models
      x_train, y_train, x_test, y_test, x_val, y_val = ttsplit_cnn(melspecs, genres, args.ctype)
      print("train, test and val size: {}, {}, {}".format(x_train.shape, x_test.shape, x_val.shape))
      
      # Start training process in keras
      input_shape = x_train[0].shape
      model, hist = train_cnn(x_train, y_train, x_val, y_val, args.ctype, 
        input_shape, args.batch_size, args.epochs)

  # Deallocate memory
  del melspecs
  del genres

  # Call garbage collect
  gc.collect()

def train_cnn(X_train, y_train, X_Val, y_val, ctype, input_shape, batch_size, epochs):
  if ctype == '1D':
    cnn = get_cnn1d(input_shape)
  elif ctype == '2D':
    cnn = get_cnn2d(input_shape)

  print("Number of parameters: %d" % cnn.count_params())
  
  # Optimizers
  sgd = keras.optimizers.SGD(lr=0.0005, momentum=0.9, decay=1e-5, nesterov=True)
  
  # Compiler for the model
  cnn.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=sgd,
    metrics=['accuracy'])

  # Fit the model
  history = cnn.fit(X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_Val, y_val))

  return cnn, history

def train_cml(x_train, y_train, x_val, y_val):
  pass

if __name__ == '__main__':
  parser = gtzan_parser()
  args = parser.parse_args()
  main(args)