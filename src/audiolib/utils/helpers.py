import numpy as np

def splitsongs(X, y, cnn_type = '1D', window = 0.1, overlap = 0.5):
  # Empty lists to hold our results
  temp_X = []
  temp_y = []

  # Get the input song array size
  xshape = X.shape[1]
  chunk = int(xshape*window)
  offset = int(chunk*overlap)

  for i, song in enumerate(X):
    spsong = [song[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
      temp_X.append(s)
      temp_y.append(y[i])

  return np.array(temp_X), np.array(temp_y)

def voting(y_true, pred):
  if y_true.shape[0] != pred.shape[0]:
    raise ValueError('Both arrays should have the same size!')

  # split the arrays in songs
  arr_size = y_true.shape[0]
  pred = np.split(pred, arr_size/augment_factor)
  y_true = np.split(y_true, arr_size/augment_factor)

  # Empty answers
  voting_truth = []
  voting_ans = []

  for x,y in zip(y_true, pred):
    voting_truth.append(mode(x)[0][0])
    voting_ans.append(mode(y)[0][0])
  
  return np.array(voting_truth), np.array(voting_ans)