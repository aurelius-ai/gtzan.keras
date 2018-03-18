import logging
import argparse

# Imports from the module we wrapped our functions
from audiolib.struct import RawAudio
from audiolib.struct import to_melspectrogram

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
  parser.add_argument('-s', '--save', 
      help='Save the RAW audios to NPY format', type=bool)
  parser.add_argument('--songs', 
      help='File path to the NPY file with songs as numpy arrays', type=str)
  parser.add_argument('--genres', 
      help='File path to the NPY file with genres one-hot encoded as numpy arrays', type=str)
  parser.add_argument('--savedir', 
      help='Path to where save the NPY files', type=str)


  return parser

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
    ds = get_features(songs)
  elif args.ctype == '1D' or '2D':
    # Convert the songs to melspectrograms
    melspecs = to_melspectrogram(songs)

  # Deallocate memory
  del songs

if __name__ == '__main__':
  parser = gtzan_parser()
  args = parser.parse_args()
  main(args)