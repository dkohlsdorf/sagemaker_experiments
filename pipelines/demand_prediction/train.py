import sys

import pickle as pkl
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *


class Forecasting(Model):

    def __init__(self, hidden=64, predict_n=30, dropout=0.1, n_labels=25):
        super().__init__()
        self.encoder = LSTM(hidden)
        self.decoder = Sequential([
            RepeatVector(predict_n),
            LSTM(hidden, return_sequences=True)
        ])
        self.predictor = Sequential([
            Dropout(0.1),
            TimeDistributed(Dense(n_labels))            
        ])

        
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        x = self.predictor(x)
        return x
    

def percentile_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


def percentile_forecasting(q):
    model = Forecasting()
    opt = Adam(learning_rate=0.01)
    model.compile(
        loss=lambda y,f: percentile_loss(q, y, f),
        optimizer=opt,
        metrics=[RootMeanSquaredError()])
    return model
    
    
def dataset(path):
    df    = pd.read_csv(path, index_col=0)
    return df
    

def create_sequences(data, t = 30):
    X = []
    y = []
    for i in range(t, len(data) - t):
        X.append(data[i-30:i])
        y.append(data[i:i+30])
    return np.array(X), np.array(y)


def save_model(path, models):
    with open(path, 'wb') as f:
        pkl.dump(models, f)

    
if __name__ == "__main__":
    path = sys.argv[1]
    out  = sys.argv[2]

    df = dataset(path)    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    models = {'scaler':scaler}
    for q in [0.1, 0.5, 0.9]:
        forecasting = percentile_forecasting(q)    
        forecasting.fit(
            X_train, y_train,
            batch_size=32, epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[]
        )
        models[f'p{q}'] = forecasting
        forecasting.summary()
    save_model(out, models)
    
