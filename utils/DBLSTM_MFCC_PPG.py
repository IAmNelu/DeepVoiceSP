import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Bidirectional, Dense, Activation, LSTM, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import datetime
import random
import math 

def get_model(hidden_units, batch_size, input_dim, output_dim, dropout) :
  model = tf.keras.models.Sequential()
  model.add(Bidirectional(LSTM(hidden_units, return_sequences=True), batch_input_shape=(batch_size, None, input_dim)))
  model.add(Dropout(dropout))
  model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
  model.add(Dropout(dropout))
  model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
  model.add(Dropout(dropout))
  model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
  model.add(Dropout(dropout))
  model.add(Dense(output_dim))
  model.add(Dropout(dropout)) 
  model.add(Activation('softmax'))
  return model

#shuffle batches of data
def shuffle_batches(X, y):
  indices = np.arange(len(X))
  random.shuffle(indices)
  new_X = [X[i] for i in indices]
  new_y = [y[i] for i in indices]
  return new_X, new_y


def padBatch(X, y, batch_size=6):
  newX = []
  newY = []
  samples = len(X)
  ts = [i.shape[0] for i in X]
  batches = math.ceil(samples / batch_size)
  for i in range(batches):
    bucket = ts[i*batch_size:(i+1)*batch_size]
    max_pb = max(bucket)
    new_shape = (max_pb, X[i].shape[1])
    new_shape_y = (max_pb, y[i].shape[1])
    for sampleX, sampleY in zip(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]):
      new_x = np.copy(sampleX)
      new_x.resize(new_shape)
      new_y = np.copy(sampleY)
      new_y.resize(new_shape_y)
      newX.append(new_x)
      newY.append(new_y)
  return newX, newY
  
def get_batches(X, y, batch_size=6):
  Xr = []
  yr = []
  samples= len(X)
  batches = math.ceil(samples / batch_size)
  for i in range(batches):
    Xr.append(np.array(X[i*batch_size:(i+1)*batch_size]))
    yr.append(np.array(y[i*batch_size:(i+1)*batch_size]))
  return Xr, yr
  
def prepare_training_data(X_train, y_train, batch_size):
    timings = [x.shape[0] for x in X_train]
    sorted_in = np.argsort(timings)
    X_train_s = [X_train[i] for i in sorted_in]
    y_train_s = [y_train[i] for i in sorted_in]
    X_train_pad, y_train_pad = padBatch(X_train_s, y_train_s, batch_size=batch_size)
    return get_batches(X_train_pad, y_train_pad)
  


class DBLSTM:
  def __init__(self, batch_size, n_mffc, hidden_units, 
               out_classes, dropout=0.3, num_epochs=10, log="train.log",
               LR=0.01, decay_rate=0.97, decay_steps=5, ch_path="", best_path="", last_path=""):
    self.initial_learning_rate = LR
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps
    self.out_classes = out_classes 
    self.epochs = num_epochs
    self.batch_size = batch_size
    self.paths = {
      "checkpoints": ch_path,
      "best": best_path,
      "last": last_path,
      "log": log
    }

    self.layer_names = ["LSTM0","LSTM1","LSTM2", "LSTM3"]
    self.model = get_model(hidden_units, batch_size, n_mffc, out_classes, dropout)
    
    self.loss_fn = CategoricalCrossentropy(from_logits=True)
    self.optimizer = Adam(self.initial_learning_rate)
 

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)

  def train_model(self, train_data, train_labels, test_data, test_labels):
    X_train, X_test, y_train, y_test = train_data, test_data, train_labels, test_labels
    
    X_train_btc, y_train_btc = prepare_training_data(X_train, y_train, self.batch_size)
    # define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_acc = tf.keras.metrics.Accuracy('test_acc', dtype=tf.float32)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = self.paths["log"] + '/mfcc2ppg/' + current_time + '/train'
    test_log_dir_loss = self.paths["log"] + '/mfcc2ppg/' + current_time +  '/test_loss'
    test_log_dir_acc = self.paths["log"] + '/mfcc2ppg/' + current_time +  '/test_accuracy'
    lr_log_dir =  self.paths["log"] + '/mfcc2ppg/' + current_time +  '/LR'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_loss_writer = tf.summary.create_file_writer(test_log_dir_loss)
    test_summary_acc_writer = tf.summary.create_file_writer(test_log_dir_acc)
    lr_summary_writer = tf.summary.create_file_writer(lr_log_dir)
    
    self.losses = []
    self.accuracies = []
    global_step = 0
    batch = 0
    
    self.model.summary()
    best_acc = 0
    for epoch in range(self.epochs):
      X, y = shuffle_batches(X_train_btc, y_train_btc)
      batch = 0
      last_loss = None
       
      for i in range(len(X)):
        global_step += 1
        batch += 1
        batch_data, batch_labels = X[i], y[i]
        
        with tf.GradientTape() as tape:
          pred = self.model(batch_data, training=True)  # Logits for this minibatch
          labels = tf.constant(batch_labels, dtype=np.float32)
          loss_value = self.loss_fn(pred, labels)
        self.losses.append(loss_value)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        train_loss(loss_value)
        with train_summary_writer.as_default():
          tf.summary.scalar('loss', train_loss.result(), step=global_step-1)
        last_loss = train_loss.result()
        train_loss.reset_states()
      eval_loss = self.evaluate_model(X_test, y_test)
      test_loss(eval_loss)
      
      with test_summary_loss_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
      template = 'Epoch {}, Loss: {}, Test Loss: {}'
      print(template.format(epoch+1, last_loss, test_loss.result()))
      train_loss.reset_states()
      test_loss.reset_states()

      # self.print_log()
      # if (acc > best_acc):
      #   best_acc = acc
      #   self.save_weights(self.best_path)




    with open(self.log_path + '.npy', 'wb') as f:
      np.save(f, self.losses)
      np.save(f, self.accuracies)
    # read with :
    # with open(self.log_path + '.npy', 'rb') as f:
    #   a = np.load(f)
    #   b = np.load(f)
  
  def evaluate_model(self, X_test, y_test):
    losses = []
    X_test_pad, y_test_pad = padBatch(X_test, y_test, batch_size=1)
    X_p, y_p =  get_batches(X_test_pad, y_test_pad, batch_size=1)

    for data, label in zip(X_p, y_p):
      pred = self.model(data, training=False)
      labels = tf.constant(label, dtype=np.float32)
      loss = tf.keras.losses.MSE(labels, pred)
      losses.append(tf.reduce_mean(loss))
    return np.mean(losses)
  
  
  def predict(self, mfcc):
    initial_eos = np.array([True]*self.batch_size)
    # reset_states(self.model, self.layer_names, initial_eos)
    ppg = self.model(np.vstack([mfcc.T[np.newaxis,:]]*self.batch_size))[0]
    # reset_states(self.model, self.layer_names, initial_eos)
    return ppg