import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Bidirectional, Dense, Activation, LSTM, Dropout
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import random
from sklearn.model_selection import train_test_split
import math 
import datetime

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
    return get_batches(X_train_pad, y_train_pad, batch_size=batch_size)
    
class DBLSTM:
  def __init__( self, dim_ppgs, dim_mceps, hidden_units, batch_size=6, lr=0.001, epochs=1, 
                dropout=0.3, decay_rate=1, decay_steps=5,
                checkpoint_path="", best_checkpoint_path="", last_checkpoint_path="", log_path="", scaler=None):
    self.batch_size = batch_size
    self.initial_lr = lr
    self.epochs = epochs
    self.dropout = dropout
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps
    self.dim_ppgs = dim_ppgs
    self.dim_mceps = dim_mceps
    self.scaler = scaler
    self.paths = {
      "checkpoints": checkpoint_path,
      "best": best_checkpoint_path,
      "last": last_checkpoint_path,
      "log": log_path
    }
    self.optimizer = Adam(lr=self.initial_lr)
    self.loss_fn = MSE
    self.model = get_model(hidden_units, batch_size, dim_ppgs, dim_mceps, dropout)


  def train_model(self, data, labels):
    #split data
    X_train, X_test, y_train, y_test = train_test_split(data, labels)
    
    X_train_btc, y_train_btc = prepare_training_data(X_train, y_train, self.batch_size)
    
    global_step = 0
    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = self.paths["log"] + '/ppg2mcep/' + current_time + '/train'
    test_log_dir = self.paths["log"] + '/ppg2mcep/' + current_time +  '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
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
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        train_loss(loss_value)
        with train_summary_writer.as_default():
          tf.summary.scalar('loss', train_loss.result(), step=global_step-1)
        last_loss = train_loss.result()
        train_loss.reset_states()
      eval_loss = self.evaluate_model(X_test, y_test)
      test_loss(eval_loss)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
      template = 'Epoch {}, Loss: {}, Test Loss: {}'
      print(template.format(epoch+1, last_loss, test_loss.result()))
      train_loss.reset_states()
      test_loss.reset_states()
      
      
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
  
  def save_model(self, path):
    self.model.save_weights(path)
  
  def load_model(self, path):
    self.model.load_weights(path)
  
  def predict(self):
    pass

  def scale_results(self, mceps):
    mceps_sc = mceps*self.scaler["std"] + self.scaler["mean"]
    return mceps_sc
  