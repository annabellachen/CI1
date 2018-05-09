# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:45:55 2018

@author: chenby
"""

from matplotlib import style
from matplotlib import pyplot as plt
style.use('ggplot')

import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    )
import datetime as dt


data = pd.read_csv('data_stocks.csv')
ts = pd.Series(data['SP500'].values, index=[dt.datetime.fromtimestamp(tt) for tt in data['DATE'].values])
ts.plot() #the distribution plot for sp500 by time series


data = data.drop(['DATE'], 1)
print(f"shape of data: ", data.shape)
print(f"columns of data: ", list(data)[:5])

data = data.values


#for the sake of normalization
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data)
data = scaler.transform(data)

X = data[:, 1:]
y = data[:, :1]
print(f"shape of X ", X.shape)
print(f"shape of y ", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"shape  ", X_train.shape)
print(f"shape  ", y_train.shape)


from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense, Activation, Flatten 
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.normalization import BatchNormalization
from sklearn.pipeline import Pipeline
from keras import optimizers


def ANN_model():
    model = Sequential()
    
    model.add(Dense(
        100,
        input_shape=(500,)))
    model.add(Activation("relu"))
    Dropout(0.5)
    model.add(Dense(20))
    model.add(Activation("relu"))
    Dropout(0.5)
    model.add(Dense(1))
    model.add(Activation("linear"))
    
    #optmzr = optimizers.SGD(lr=0.002, momentum=0.2, decay=0.009, nesterov=False)
    #optmzr = optimizers.Adadelta(lr=.02, rho=0.95, epsilon=None, decay=0.009)
    optmzr = optimizers.Adam(lr=.02, beta_1=0.1, beta_2=0.11, epsilon=None, decay=0.009, amsgrad=False)
    #optmzr = optimizers.RMSprop(lr=0.02, rho=0.9, epsilon=None, decay=0.009)
    model.compile(optimizer=optmzr,loss='mse')
    return model


ann_model = ANN_model()

ann_model_history = ann_model.fit(X_train, y_train, validation_split=0.1, epochs=10)


ann_midResult = ann_model.predict(X_test[:10])
from sklearn.metrics import mean_squared_error
ann_model_rmse = np.sqrt(mean_squared_error(ann_midResult, y_test[:10]))

#helper func to plot history
def plotHistory(history, title, xTitle, yTitle):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel(yTitle)
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
seq = 50

# only take sp500 column
def prepare_lstm_data(data, seq_len):
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)

    row = round(0.8 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]
def LSTM_model(optim):
    model = Sequential()
    
    model.add(LSTM(
        1,
        input_shape=(seq, 1),
        return_sequences=True))
    
    model.add(LSTM(
        100,
        return_sequences=False))
    
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(optimizer=optim,loss='mse')
    return model
X_train2, y_train2, X_test2, y_test2 = prepare_lstm_data(y, seq)

lstm_model = LSTM_model("adam")
lstm_model_history = lstm_model.fit(X_train2, y_train2, batch_size=128, epochs=3, validation_split=0.05)
def plot_predict(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.title('LSTM model loss')
    plt.legend()
    plt.show()
    #plot train and validation for LSTM model
lstm_model_preds = lstm_model.predict(X_test2)
lstm_model_rmse = np.sqrt(mean_squared_error(lstm_model_preds, y_test2))
plot_predict(lstm_model_preds, y_test2)
plotHistory(lstm_model_history, "LSTM Model", "loss", "epochs")
plotHistory(ann_model_history, "ANN Model", "lose", "epochs")
from neupy import algorithms, estimators, environment
environment.reproducible()
grnn_model = algorithms.GRNN(std=0.1, verbose=False)
grnn_model.train(X_train, y_train)
grnn_predicted = grnn_model.predict(X_test)
grnn_model_rmse = estimators.rmse(grnn_predicted, y_test)
print('ANN RMSE', ann_model_rmse)
print('LSTM RMSE', lstm_model_rmse)
print('GRNN RMSE', grnn_model_rmse)