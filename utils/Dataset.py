import numpy as np

class DataSet(object):
  def __init__(self, data, labels, batch_size=6):
    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = data.shape[0]
    # required for next_sequence_batch
    self._batch_size = batch_size
    # List for the indexes of the sentences in the current batch
    self._indexes_in_epoch = list(range(batch_size))
    # Array for the frame indexes in the sentences for the current batch
    # Values are always a multiple of seqence_size
    self._index_in_sentences = np.zeros([batch_size], dtype=int)
    
  @property
  def data(self):
    return self._data
      
  @property
  def labels(self):
    return self._labels
  
  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
      
  @property
  def batch_size(self):
    return self._batch_size
  
  def reset_epoch(self, batch_size):
    self._batch_size = batch_size
    self._indexes_in_epoch = list(range(batch_size))
    self._index_in_sentences = np.zeros([self._batch_size], dtype=int)
    # Shuffle the data
    perm = np.arange(len(self._data))
    np.random.shuffle(perm)
    self._data = self._data[perm]
    self._labels = self._labels[perm]
  
  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(len(self._data))
      np.random.shuffle(perm)
      self._data = self._data[perm]
      self._labels = self._labels[perm]
      start = 0
      self._index_in_epoch = batch_size
    
    end = self._index_in_epoch
    return self._data[start:end], self._labels[start:end]
      
  def next_sequence_batch(self, seq_length):
    # Boolean array. If _end_of_sentence[i] = False, the state should be
    # preserved in an RNN structure. Otherwise the state for sentence i
    # should be reset
    end_of_sentence = np.zeros([self._batch_size], dtype=bool)        
    starts = np.copy(self._index_in_sentences)
    for i in range(self._batch_size):
      self._index_in_sentences[i] += seq_length
      # If the end of this sentence is reached ...
      if (self._index_in_sentences[i] > self.data[self._indexes_in_epoch[i]].shape[0]):
        # ..set the end_of_sentence flag
        end_of_sentence[i] = True
        # ..fetch a new sentence (the sentence after the furthest sentence in this epoch)
        new_index = np.max(self._indexes_in_epoch) + 1
        # Check if there are no more sentences in this epoch
        if (new_index >= self._num_examples):
          #print("End of epoch")
          self._epochs_completed += 1
          # Reset epoch
          self.reset_epoch(self._batch_size)                  
          # Return Nones to signal the training loop
          return None, None, None
        # Otherwise, reset the index in sentence and set the new index in epoch
        self._indexes_in_epoch[i] = new_index
        self._index_in_sentences[i] = seq_length
        starts[i] = 0
    
    # Get for each sentence in the batch the sequence from 
    # index_in_sentence to starts + seq_length
    batch_data = [
      self._data[self._indexes_in_epoch[i]][starts[i]:starts[i]+seq_length] 
      for i in range(self._batch_size)
    ]
    # Same for the labels
    batch_labels = [
      self._labels[self._indexes_in_epoch[i]][starts[i]:starts[i]+seq_length] 
      for i in range(self._batch_size)
    ]
    return np.array(batch_data), np.array(batch_labels), np.array(end_of_sentence)
      
  def set_as_sequences(self, sequence_length):
    num_sequences = len(self._data) - sequence_length
    sequence_data = [self._data[i:(i+sequence_length)] for i in range(0, num_sequences)]
    sequence_labels = [self._labels[i:(i+sequence_length)] for i in range(0, num_sequences)]
    return sequence_data, sequence_labels

def get_avarage_size_dataset(dataset, sequence_length=20, average_size=2, verbose=False):
  counts = []
  for i in range(average_size):
    ap, bp = None, None
    count = 0
    while True:
      a, b, _ = dataset.next_sequence_batch(sequence_length)
      count+=1
      
      if a is None:
        counts.append(count)
        break
      else:
        if np.isnan(a).any() or np.isnan(b).any():
          print(f"bastardo {a} {b}")
      ap, bp = a, b
  
  res = np.sum(np.array(counts)) / average_size
  if verbose:
    print(f"With {average_size} runs average the dataset is made by {res:.2f} sets of sequences.")
  return res