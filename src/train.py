import argparse

# Imports from the module we wrapped our functions
from audiolib.struct import RawAudio

# List with all the genres available on the GTZAN dataset
classes = [
  'metal', 'disco', 'classical', 'hiphop', 'jazz',
  'country', 'pop', 'blues', 'reggae', 'rock'
]

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
  

  return parser

def main(args):
  # Get the GTZAN dataset from raw audio
  audio = RawAudio('RAW', file_path='../dataset/GTZAN/', classes=classes)
  songs, genres = audio.get_data()
  RawAudio.save_data(songs = songs, genres = genres, file_path = '../dataset/GTZAN/')

  print(songs.shape, genres.shape)

if __name__ == '__main__':
  parser = gtzan_parser()
  args = parser.parse_args()
  main(args)