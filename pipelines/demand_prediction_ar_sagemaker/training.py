import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.models import * 
from tensorflow.keras.layers import * 

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import * 
from tensorflow.keras.metrics import * 

from s3fs.core import S3FileSystem


class AutoRegressive(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = LSTMCell(units)
    self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
    self.dense = Dense(25)
    
  def warmup(self, inputs):
    x, *state = self.lstm_rnn(inputs)
    prediction = self.dense(x)
    return prediction, state

  def call(self, inputs, training=None):
    predictions = []
    prediction, state = self.warmup(inputs)
    predictions.append(prediction)
    
    for n in range(1, self.out_steps):
      x = prediction
      x, state = self.lstm_cell(x, states=state, training=training)
      prediction = self.dense(x)
      predictions.append(prediction)
    
    predictions = tf.stack(predictions)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions
    

def decode_fn(record_bytes):
  return tf.io.parse_single_example(record_bytes, {"demand": tf.io.FixedLenFeature([], dtype=tf.float32),})


def data(path, History=4, Horizon=4):
    X = [] 
    print(f">>> {path}")
    for batch in tf.data.TFRecordDataset([path]).map(decode_fn):
        X.append(batch)
        print(f" >>> {batch}")

    inputs = []
    predictions = []
    for i in range(History, len(X)-Horizon):
        x = X[i-History:i]
        y = X[i:i+Horizon]
        inputs.append(x)
        predictions.append(y)
    x, y = np.array(inputs), np.array(predictions)
    print(f" >>>> {x.shape} {y.shape}")
    return x, y     


def train(X, y, latent=32, History=4):
    model = AutoRegressive(latent, History)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[RootMeanSquaredError()])
    model.fit(X, y, batch_size=8, epochs=250)
    return model


def write(model):
    model.save('./tensorflow/', save_format='tf')


def parse_args():
    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument(
        "--train_data",
        type=str,
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args  = parse_args()
    X, y  = data(args.train_data)
    model = train(X, y)
    write(model)

