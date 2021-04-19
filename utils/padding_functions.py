import numpy as np

def remove_silence(X, cl=None, padd=1):
  snello = {}
  for key, v in X.items():
    x = v['mfcc']
    y = v['y']
    if len(x) != len(y):
      print("MIS MATCH X-Y")
    classes = np.argmax(y, axis=1) #go from one hot to index encoding
    for i, c in enumerate(classes):
      if c != cl:
        init = i
        break
    for i, c in enumerate(classes[::-1]):
      if c != cl:
        end = len(classes) - i - 1
        break
    init -= padd # add silence equal to the number of padding requested 
    end += padd # add silence equal to the number of padding requested 
    # print(f"{init}-{end}")
    snello[key] = {'mfcc':x[init:end+1], 'y':y[init:end+1]} 
  return snello


def pad_last(sentence, target_length):
  """
  Pad the mfcc of a sentence to a given target length using the last frame.
  This last frame is considered to be silence in the mfcc files, therefore
  acting as a neutral element.
  
  Args:
      sentence (numpy.ndarray):
          An ndarray containing mfcc data.
          Its shape is [sentence_length, coefficients]
      target_length (int):
          The total length after padding. If this is smaller than the 
          sentence length, an error will be raised.
          
  Returns:
      numpy.ndarray:
          An ndarray containing mfcc data padded to target_length.
  """
  
  assert len(sentence) <= target_length, "Can't pad sentence of length %i to length %i"%(len(sentence), target_length)
  last = sentence[-1]
  pad = [last for i in range(0, target_length - len(sentence))]
  
  return np.append(sentence, pad, 0)

def pad_to_sequence_length(sentence, seq_length):
  """
  Pad a sentence with its last frames such that the length of the sequence
  becomes a multiple of seq_length.
  
  This method can be used for padding labels and data.
  
  Args:
      sentence (numpy.ndarray):
          An ndarray containing the data that should be padded.
      seq_length (int):
          The length of one sequence.
  
  Returns:
      numpy.ndarray:
          The same sentence as the input sentence, padded to a multpile of 
          seq_lengths
  """
  
  padding_length = seq_length - (sentence.shape[0] % seq_length)
  target_length = sentence.shape[0] + padding_length
  return pad_last(sentence, target_length)

def pad_data(dictionay_data, seq_length=20):
  """
  Take a data dictionay having as key the speaker and as value a dictionary with
  at least the following keys: "mfcc" and "y" (labels)

  Args:
    dictionay_data (dict):
      Dictionay with all data as described in the introduction
    seq_length (int):
      The length of one sequence.
    
    Returns:
      List:
        list of Xp (mfcc padded) and label (y
        padded)

  """
  Xp = []
  labels = []
  for speaker, data in dictionay_data.items():
    x = data['mfcc']
    y = data['y']
    if x.shape[0]==0 or y.shape[0]==0:
      pass
    else:
      x_p = pad_to_sequence_length(x, seq_length)
      y_p = pad_to_sequence_length(y, seq_length)
      Xp.append(x_p)
      labels.append(y_p)

  return Xp, labels
