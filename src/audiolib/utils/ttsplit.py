from . import *
from sklearn.model_selection import train_test_split

# @Function: ttsplit_cml
# @Description: Split in train and test a set of audio
# features from the GTZAN dataset
def ttsplit_cml(features):
  pass

# @Function: ttsplit_cnn
# @Description: Split in train and test a set of melspectrograms
# using non-overlapping windows
def ttsplit_cnn(melspec, genres, ctype):
  # Split the dataset into training and test
  X_train, X_test, y_train, y_test = train_test_split(
    melspec, genres, test_size=0.1, stratify=genres)
    
  # Split training set into training and validation
  X_train, X_Val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=1/6, stratify=y_train)

  return X_train, y_train, X_test, y_test, X_Val, y_val