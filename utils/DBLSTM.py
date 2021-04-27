import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Bidirectional, Dense, Activation, LSTM, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def get_data_array(dataset, sequence_length=20):
  X_train = []
  y_train = []
  eoses = []
  while True:
    batch_data, batch_labels, eos = dataset.next_sequence_batch(sequence_length)
    if batch_data is None:
      break
    X_train.append(batch_data)
    y_train.append(batch_labels)
    eoses.append(eos)
  X_train = np.array(X_train)
  y_train = np.array(y_train)
  eoses = np.array(eoses)
  return X_train, y_train, eoses
  
def reset_layer(layer, eos): #resets inner states
  for i in range(len(layer.states)):
    current_state = layer.states[i].numpy()
    current_state[eos] = 0
    layer.states[i].assign(current_state)


def reset_states(model, layers, eos):
  for nlayer in layers:
    layer = model.get_layer(nlayer) # get one lyer, LSTM or Bidirectional
    try:# LSTM
      reset_layer(layer, eos) 
    except:#bidirectional
      reset_layer(layer.forward_layer, eos) 
      reset_layer(layer.backward_layer, eos)


class _LSTM:
  def __init__(self, batch_size, sequence_length, n_mffc, hidden_units, 
               out_classes, dropout=0.3, num_epochs=10, log="train.log",
               LR=0.01, ch_path=None):
    self.LR = LR
    self.loss_fn = CategoricalCrossentropy(from_logits=True)
    self.optimizer = Adam(LR)
    self.epochs = num_epochs
    self.sentence_length = sequence_length
    self.batch_size = batch_size
    self.ch_path = ch_path
    self.layer_names = ["LSTM0","LSTM1","LSTM2", "LSTM3"]
    model = tf.keras.models.Sequential()
    model.add(LSTM(hidden_units, return_sequences=True, stateful=True, name= self.layer_names[0], batch_input_shape=(batch_size, sequence_length, n_mffc)))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_units, return_sequences=True, stateful=True, name= self.layer_names[1]))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_units, return_sequences=True, stateful=True, name= self.layer_names[2]))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_units, return_sequences=True, stateful=True, name= self.layer_names[3]))
    model.add(Dropout(dropout))
    model.add(Dense(out_classes))
    model.add(Dropout(dropout))
    model.add(Activation('softmax'))
    self.out_classes = out_classes 
    self.model  = model
    self.logs = []
    self.log_path = log

  def evaluate_model(self, test_dataset):
    output_labels = []
    ground_truth_labels = []
    initial_eos = np.array([True]*self.batch_size)
    reset_states(self.model, self.layer_names, initial_eos) #init states at zero

    X, y, eoses = get_data_array(test_dataset, sequence_length=self.sentence_length)

    for i in range(len(X)):
      batch_data, batch_labels, eos = X[i], y[i], eoses[i]
      
      pred = self.model(batch_data, training=False)  # Logits for this minibatch

      labels = tf.constant(batch_labels, dtype=np.float32)

      #transform prediction to array of most probable phoneme index
      labels_pred = tf.reshape(tf.math.argmax(pred, axis=2), [pred.shape[0]*pred.shape[1]])
      #get phoneme index for the ground truth
      labels_truth = tf.reshape(tf.math.argmax(batch_labels, axis=2), [batch_labels.shape[0] * batch_labels.shape[1]])

      output_labels= np.concatenate((output_labels, labels_pred.numpy()))
      ground_truth_labels= np.concatenate((ground_truth_labels,labels_truth.numpy()))

      if True in eos:
        reset_states(self.model, self.layer_names, eos)

    correct_labels = np.count_nonzero(output_labels==ground_truth_labels)
    accuracy = correct_labels/len(output_labels)
    return output_labels, ground_truth_labels, accuracy
  def train_model(self, train_dataset, test_dataset):
    self.losses = []
    self.accuracies = []
    global_step = 0
    batch = 0
    print("===START TRAINING ===\n", f"N EPOCHS: {self.epochs}")
    print(f"LR: {self.LR}")
    print(f"=== TIME {time.asctime(time.localtime())}===")
    self.model.summary()
    start_training_time = time.time()
    for epoch in range(self.epochs):
      start_epoch_time = time.time()
      initial_eos = np.array([True]*self.batch_size)
      reset_states(self.model, self.layer_names, initial_eos) #initi states at zero
      print(f"EPOCH {epoch} == {time.asctime(time.localtime())}==")
      batch = 0
      X, y, eoses = get_data_array(train_dataset, sequence_length=self.sentence_length)
      for i in range(len(X)):
        global_step += 1
        batch += 1
        batch_data, batch_labels, eos = X[i], y[i], eoses[i]
        with tf.GradientTape() as tape:
          pred = self.model(batch_data, training=True)  # Logits for this minibatch
          labels = tf.constant(batch_labels, dtype=np.float32)
          # print(pred.shape)
          # print(labels.shape)
          loss_value = self.loss_fn(pred, labels)
        self.losses.append(loss_value)
        if True in eos:
          reset_states(self.model, self.layer_names, eos)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # print(f"Step {i} - Loss: {float(self.losses[-1]):.4f}")
        # if i % 23 == 0:
        #   # print(pred[i%6].shape)
        #   # print(labels[i%6].shape)
        #   plot_batch(pred[i%6], labels[i%6])
          
        if i % 400 == 0:
          print(f"TRAINING LOSS (for one batch) at step {i}/{len(X)}:{float(self.losses[-1]):.4f}")
          print(f"Seen so far: {((global_step + 1) * self.batch_size)} samples")
          print("")
        
      # break
      end_epoch = time.time()
      print(f"EPOCH {epoch} COMPLETED == {time.asctime(time.localtime())}==")
      time_ellapsed  = (end_epoch - start_epoch_time)
      print(f"== TIME TRAINING EPOCH {time_ellapsed} seconds ==")
      predicted, ground_truth , acc = self.evaluate_model(test_dataset)
     
      cf_np = confusion_matrix(ground_truth, predicted, labels=range(0,self.out_classes))
      cf_lst = [str(r) + "\n" for r in cf_np]
      print(f"== ACCURACY ON TEST-SET {acc * 100:.2f}% ==")
      mx=""
      for s in cf_lst: 
        mx += s
      print(mx)
    end_training = time.time()
    print("===FINISHED TRAINING ===\n", f"N EPOCHS: {self.epochs}\n", f"LR: {self.LR}")
    print(f"=== TIME {time.asctime(time.localtime())}===")  
    time_ellapsed  = (end_training - start_training_time)   
    print(f"== TIME TRAINING ALL {time_ellapsed} seconds ====\n\n\n")
    
  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)

class DBLSTM:
  def __init__(self, batch_size, sequence_length, n_mffc, hidden_units, 
               out_classes, dropout=0.3, num_epochs=10, log="train.log",
               LR=0.01, decay_rate=0.97, decay_steps=5, ch_path=None):
    self.initial_learning_rate = LR
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps
    self.LR = LR
    self.loss_fn = CategoricalCrossentropy(from_logits=True)
    self.optimizer = Adam(self.LR)
    self.epochs = num_epochs
    self.sentence_length = sequence_length
    self.batch_size = batch_size
    self.ch_path = ch_path
    self.layer_names = ["LSTM0","LSTM1","LSTM2", "LSTM3"]
    model = tf.keras.models.Sequential()
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, stateful=True, name="__" + self.layer_names[0]), name=self.layer_names[0], batch_input_shape=(batch_size, sequence_length, n_mffc)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, stateful=True, name="__" + self.layer_names[1]), name=self.layer_names[1]))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, stateful=True, name="__" + self.layer_names[2]), name=self.layer_names[2]))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, stateful=True, name="__" + self.layer_names[3]), name=self.layer_names[3]))
    model.add(Dropout(dropout))
    model.add(Dense(out_classes))
    model.add(Dropout(dropout))
    model.add(Activation('softmax'))
    self.out_classes = out_classes 
    self.model  = model
    self.logs = []
    self.log_path = log
 
  def decayed_learning_rate(self, step):
    return self.initial_learning_rate * self.decay_rate**(step / self.decay_steps)

  def print_log(self):
    if len(self.logs) == 0:
      return  
    with open(self.log_path, 'a') as f:
      f.writelines(self.logs)
      self.logs = []

  def evaluate_model(self, test_dataset):
    output_labels = []
    ground_truth_labels = []
    initial_eos = np.array([True]*self.batch_size)
    reset_states(self.model, self.layer_names, initial_eos) #init states at zero

    X, y, eoses = get_data_array(test_dataset, sequence_length=self.sentence_length)

    for i in range(len(X)):
      batch_data, batch_labels, eos = X[i], y[i], eoses[i]
      
      pred = self.model(batch_data, training=False)  # Logits for this minibatch

      labels = tf.constant(batch_labels, dtype=np.float32)

      #transform prediction to array of most probable phoneme index
      labels_pred = tf.reshape(tf.math.argmax(pred, axis=2), [pred.shape[0]*pred.shape[1]])
      #get phoneme index for the ground truth
      labels_truth = tf.reshape(tf.math.argmax(batch_labels, axis=2), [batch_labels.shape[0] * batch_labels.shape[1]])

      output_labels= np.concatenate((output_labels, labels_pred.numpy()))
      ground_truth_labels= np.concatenate((ground_truth_labels,labels_truth.numpy()))

      if True in eos:
        reset_states(self.model, self.layer_names, eos)

    correct_labels = np.count_nonzero(output_labels==ground_truth_labels)
    accuracy = correct_labels/len(output_labels)
    return output_labels, ground_truth_labels, accuracy
  #output is: vector of predicted phoneme key, vector of true phoneme key, total accuracy

  def evaluate_x_batches(self, dataset, n_batches):
    output_labels = []
    ground_truth_labels = []
    initial_eos = np.array([True]*self.batch_size)
    reset_states(self.model, self.layer_names, initial_eos) #init states at zero

    X, y, eoses = get_data_array(dataset, sequence_length=self.sentence_length)

    for i in range(min(len(X), n_batches)):
      batch_data, batch_labels, eos = X[i], y[i], eoses[i]
      
      pred = self.model(batch_data, training=False)  # Logits for this minibatch

      labels = tf.constant(batch_labels, dtype=np.float32)

      #transform prediction to array of most probable phoneme index
      labels_pred = tf.reshape(tf.math.argmax(pred, axis=2), [pred.shape[0]*pred.shape[1]])
      #get phoneme index for the ground truth
      labels_truth = tf.reshape(tf.math.argmax(batch_labels, axis=2), [batch_labels.shape[0] * batch_labels.shape[1]])

      output_labels= np.concatenate((output_labels, labels_pred.numpy()))
      ground_truth_labels= np.concatenate((ground_truth_labels,labels_truth.numpy()))

      if True in eos:
        reset_states(self.model, self.layer_names, eos)

    correct_labels = np.count_nonzero(output_labels==ground_truth_labels)
    accuracy = correct_labels/len(output_labels)
    return output_labels, ground_truth_labels, accuracy

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)

  def train_model(self, train_dataset, test_dataset):
    
    self.losses = []
    self.accuracies = []
    global_step = 0
    batch = 0
    self.logs += ["===START TRAINING ===\n", f"N EPOCHS: {self.epochs}\n", f"LR: {self.LR}\n"]
    self.model.summary(print_fn=lambda x: self.logs.append(x+"\n"))
    self.logs +=[f"=== TIME {time.asctime(time.localtime())}===\n"]
    start_training_time = time.time()
    self.print_log()
    for epoch in range(self.epochs):
      start_epoch_time = time.time()
      initial_eos = np.array([True]*self.batch_size)
      reset_states(self.model, self.layer_names, initial_eos) #initi states at zero
      self.logs.append(f"EPOCH {epoch} == {time.asctime(time.localtime())}==\n")
      # print(f"Epoch {epoch}")
      batch = 0
      X, y, eoses = get_data_array(train_dataset, sequence_length=self.sentence_length)
      for i in range(len(X)):
        global_step += 1
        batch += 1
        batch_data, batch_labels, eos = X[i], y[i], eoses[i]
        
        with tf.GradientTape() as tape:
          pred = self.model(batch_data, training=True)  # Logits for this minibatch
          labels = tf.constant(batch_labels, dtype=np.float32)
          loss_value = self.loss_fn(pred, labels)
        self.losses.append(loss_value)
        if True in eos:
          reset_states(self.model, self.layer_names, eos)
        
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
      
        if i % 400 == 0:
          self.logs.append(f"TRAINING LOSS (for one batch) at step {i}/{len(X)}:{float(self.losses[-1]):.4f}\n")
          self.logs.append(f"Seen so far: {((global_step + 1) * self.batch_size)} samples\n")
          self.print_log()

      self.LR = self.decayed_learning_rate(epoch)
      # print(self.LR)
      self.logs.append(f"NEW LR: {self.LR}=======\n")

      end_epoch = time.time()
      self.logs.append(f"EPOCH {epoch} COMPLETED == {time.asctime(time.localtime())}==\n")
      time_ellapsed  = (end_epoch - start_epoch_time)
      self.model.save_weights(self.ch_path.format(epoch=epoch))
      self.logs.append(f"== TIME TRAINING EPOCH {time_ellapsed} seconds ==\n")
      predicted, ground_truth , acc = self.evaluate_model(test_dataset)
     
      cf_np = confusion_matrix(ground_truth, predicted, labels=range(0,self.out_classes))
      cf_lst = [str(r) + "\n" for r in cf_np]
      self.logs.append(f"== ACCURACY ON TEST-SET {acc * 100:.2f}% ==\n")
      self.logs.append("=== CONFUSION MATRIX ===")
      self.logs += cf_lst
      self.logs.append("=== CONFUSION MATRIX FINISH ===")
      self.print_log()

    end_training = time.time()
    self.logs += ["===FINISHED TRAINING ===\n", f"N EPOCHS: {self.epochs}\n", f"LR: {self.LR}\n"]
    self.logs +=[f"=== TIME {time.asctime(time.localtime())}===\n"]
    time_ellapsed  = (end_training - start_training_time)
    self.logs.append(f"== TIME TRAINING ALL {time_ellapsed} seconds ====\n\n\n")
    self.print_log()
    self.losses = np.array(self.losses)
    self.accuracies = np.array(self.accuracies)


    with open(self.log_path + '.npy', 'wb') as f:
      np.save(f, self.losses)
      np.save(f, self.accuracies)
    # read with :
    # with open(self.log_path + '.npy', 'rb') as f:
    #   a = np.load(f)
    #   b = np.load(f)
