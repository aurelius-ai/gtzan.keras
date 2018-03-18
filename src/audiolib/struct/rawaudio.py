import os
import keras
import librosa
import numpy as np

# @Class: RawAudio
#
# @Description: 
#  Class to read .au files and store then in memory
#
# @Usage:
#  RawAudio('NPY', songs='', genres='')
#  RawAudio('RAW', file_path='', classes='')
#
# @Constructor Parameters:
# * songs: Path to the npy song file
# * genres: Path to the npy genre file
# * file_path: Root of the GTZAN folder
# * classes: classes to read in the GTZAN folders
class RawAudio(object):
  def __init__(self, dtype, **kwargs):

    self.dtype = dtype

    # faster alternative to read files: read from npy file
    # You can use the function "save_data" to save the np arrays to npy
    if self.dtype == 'NPY':
      # constructor parameters
      songs_file_path = kwargs.get('songs', None)
      genres_file_path = kwargs.get('genres', None)

      self.songs = np.load(songs_file_path)
      self.genres = np.load(genres_file_path)

    # Prepare parameters to Iterate over a directory
    # and get au the .au files in memory
    elif self.dtype == 'RAW':
      # constructor parameters
      file_path = kwargs.get('file_path', None)
      list_genre = kwargs.get('classes', None)

      # Constants
      self.song_samples = 660000
      self.file_path = file_path
      
      # Create a dictionary with genre list and the mapping to integers
      self.list_genre = list_genre
      self.genres = {v:k for k, v in enumerate(self.list_genre)}

    # Throw an exception because dtype is invalid for this app
    else:
      raise Exception('invalid dtype')
   
  # @Method: getdata
  # @Description:
  #  Retrieve data from NPY file or .au files and return then as numpy arrays
  # @Output: Songs, Genres
  def get_data(self):
    # Get data from NPY files
    if self.dtype == 'NPY':
      return self.songs, self.genres

    # Structure for the array of songs
    song_data = []
    genre_data = []
        
    # Read files from the folders
    for x,_ in self.genres.items():
      for root, subdirs, files in os.walk(self.file_path + x):
        for file in files:
          # Read the audio file
            file_name = self.file_path + x + "/" + file
            print(file_name)
            signal, sr = librosa.load(file_name)
          
            # Calculate the melspectrogram of the audio and use log scale
            raw = signal[:self.song_samples]
            
            # Append the result to the data structure
            song_data.append(raw)
            genre_data.append(self.genres[x])
    return np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))

  # @Method: save_data
  # @Description:
  #  Save an array of songs and genres to npy format
  @staticmethod
  def save_data(songs, genres, file_path):
    # Save the files to npy
    np.save(file_path + 'songs.npy', songs)
    np.save(file_path + 'genres.npy', genres)