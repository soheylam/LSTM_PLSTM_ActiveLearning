import numpy as np
import pandas as pd
import pdb
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import matplotlib.pyplot as plt  
import tensorflow.keras.backend as K
     

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.keras.utils.set_random_seed(812)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  if dropnan:
    agg.dropna(inplace=True)
  return pd.DataFrame(agg.astype('float32'))


def train_valid_test_split(data, hours_of_history, hours_to_predict, parameters_included):
  data_train_valid = data.iloc[:19200,:]   
  data_test = data.iloc[19200:,:] 
  data_train_valid.dropna(inplace=True)
  data_test.dropna(inplace=True)
  data_valid, data_train = train_test_split(data_train_valid, test_size=0.8, shuffle= False) 
  return data_train.values, data_valid.values, data_test.values


def prepare_data(model_id, hours_of_history, hours_to_predict, parameters_included, noise_scale=0.1):
  data = pd.read_csv('precip_streamflow_data.csv')

  scaler = MinMaxScaler()
  scaler.fit(data.iloc[:19200,:]) 
  q_max = np.max(data.iloc[:19200,1]) 
  q_min = np.min(data.iloc[:19200,1])
  data_scaled = scaler.transform(data)

  # data split
  data_sequence = series_to_supervised(data_scaled, hours_of_history, hours_to_predict)
  data_train, data_valid, data_test = train_valid_test_split(data_sequence, hours_of_history, hours_to_predict, parameters_included)

  train_x_rainfall = data_train[:,0::2].reshape(-1, hours_of_history+hours_to_predict, 1)
  train_discharge = data_train[:,1::2].reshape(-1, hours_of_history+hours_to_predict, 1)
  train_x_discharge = train_discharge[:,:hours_of_history,:]
  train_y = train_discharge[:,hours_of_history:,:]

  valid_x_rainfall = data_valid[:,0::2].reshape(-1, hours_of_history+hours_to_predict, 1)
  valid_discharge = data_valid[:,1::2].reshape(-1, hours_of_history+hours_to_predict, 1)
  valid_x_discharge = valid_discharge[:,:hours_of_history,:]
  valid_y = valid_discharge[:,hours_of_history:,:]

  test_x_rainfall = data_test[:,0::2].reshape(-1, hours_of_history+hours_to_predict, 1)
  test_discharge = data_test[:,1::2].reshape(-1, hours_of_history+hours_to_predict, 1)
  test_x_discharge = test_discharge[:,:hours_of_history,:]
  test_y = test_discharge[:,hours_of_history:,:]

  return [train_x_discharge, train_x_rainfall], train_y, [valid_x_discharge, valid_x_rainfall], valid_y, [test_x_discharge, test_x_rainfall], test_y, q_max, q_min


def asymmetric_peak_loss(y_true, y_pred):
    T = 0.45  # threshold
    F = 3.0   # asymmetry factor
    squared_error = K.square(y_true - y_pred)
    mse = K.mean(squared_error)
    # indicators:
    above_threshold = K.cast(y_true > T, dtype='float32')
    underestimation = K.cast(y_true > y_pred, dtype='float32')
    penalty = K.mean(squared_error * F * above_threshold * underestimation)
    return mse + penalty

def EDLSTM(hours_of_history, hours_to_predict, parameters_included):
  input_1 = Input(shape=(hours_of_history, 1), name='LSTM1_input') 
  LSTM1 = LSTM(256, return_sequences=False)(input_1)
  input_2 = Input(shape=((hours_of_history+hours_to_predict), 1), name='LSTM2_input') 
  LSTM2 = LSTM(256, return_sequences=False, recurrent_dropout=0.45)(input_2)
  x = concatenate([LSTM1, LSTM2]) 
  x = RepeatVector(hours_to_predict)(x) 
  x = LSTM(512, return_sequences=True)(x)
  dim_dense=[512, 256, 128, 64]

  for dim in dim_dense:
    x = TimeDistributed(Dense(dim, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.2))(x)   
  main_out = TimeDistributed(Dense(1, activation='relu'))(x) 
  main_out = Flatten()(main_out)
  model = Model(inputs=[input_1, input_2], outputs=main_out)
  return model

def nse(y_true, y_pred):
  return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)
###################################################################################################################
def main():
  
  # parameters
  model_id = 'your_model_id'
  hours_to_predict = 12
  hours_of_history = 60
  parameters_included = 2

  batch_size = 64      
  lr = 0.0001          ####   0.001        
  epochs = 300   
  test_name = 'your_path'+model_id+'_model1_APL'


  
  x_train, y_train, x_valid, y_valid, x_test, y_test, q_max, q_min = prepare_data(model_id, hours_of_history, hours_to_predict, parameters_included)
  model1 = EDLSTM(hours_of_history, hours_to_predict, parameters_included)
  
  earlystoping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')   ##patience=10
  checkpoint = ModelCheckpoint(test_name+'model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  optimizer = Adam(learning_rate=lr)

  peak_threshold = 0.45
  asymmetry_factor = 3.0 
  model1.compile(optimizer=optimizer, loss=asymmetric_peak_loss) 
  history = model1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(x_valid, y_valid), callbacks=[earlystoping, checkpoint], verbose=1)
   
  # save training loss
  loss_train = history.history['loss']
  loss_valid = history.history['val_loss']
  loss_train = pd.DataFrame({'TrainLoss':loss_train})
  loss_valid = pd.DataFrame({'TestLoss':loss_valid})
  LossEpoches = pd.concat([loss_train, loss_valid], axis=1)

  model1.load_weights(test_name+'model.h5')
  y_model_scaled = model1.predict(x_test)
  y_model = y_model_scaled*(q_max-q_min)+q_min
  y_test = y_test*(q_max-q_min)+q_min
  

  
  # hourly evaluation
  NSEs=[]
  for x in range(0, 12):
    y_pred = y_model[:,x]
    y_True = y_test[:,x]
    NSEs.append(nse(y_True[:,0],y_pred))
  NSEs=pd.DataFrame(NSEs)
  NSEs.columns = ['NSE_Test']
  

  # Convert to DataFrame for plotting and saving
  lead_times = np.arange(1, hours_to_predict + 1)
  NSE_df = pd.DataFrame({
      "Lead Time (hr)": lead_times,
      "NSE": np.array(NSEs).flatten()  # <-- flatten to 1D
  })
  
if __name__ == "__main__":
  main()

