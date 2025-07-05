import numpy as np
import pandas as pd
import random
from scipy.stats import norm, uniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

# Load the distributions information from a file (adjust path as necessary)
distribution_info = pd.read_csv('distribution_info.csv')

# Convert the 'parameters' column from string to tuple
distribution_info['parameters'] = distribution_info['parameters'].apply(eval)

# Define a function to get the distribution parameters for a given forecast file and category
def get_distribution_params(forecast_file, category):
    row = distribution_info[(distribution_info['forecast_file'] == forecast_file) & 
                            (distribution_info['category'] == category)]
    if not row.empty:
        dist = row['best_distribution'].values[0]
        params = row['parameters'].values[0]
        return dist, params
    return None, None

def categorize_rainfall(rainfall):
    """Categorize rainfall into defined categories."""
    if rainfall == 0:
        return 'No Observed Precip'
    elif 0 < rainfall < 1:
        return 'Very low_0 < rainfall < 1'
    elif 1 <= rainfall < 5:
        return 'Low_1 <= rainfall < 5'
    elif 5 <= rainfall < 10:
        return 'Moderate_5 <= rainfall < 10'
    elif rainfall >= 10:
        return 'High_rainfall >= 10'
    return None

def sample_noise_from_distributions(dist, params):
    """Sample noise from the distribution based on the distribution type and parameters."""
    if dist == 'norm':
        return norm.rvs(loc=params[0], scale=params[1])
    elif dist == 'uniform':
        return uniform.rvs(loc=params[0], scale=params[1])
    else:
        raise ValueError(f"Unsupported distribution type: {dist}")

def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    """Convert series to supervised learning format."""
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
    """Split data into train, validation, and test sets (40 events;19200 values are for training and the remaining is the testset )."""
    data_train_valid = data.iloc[:19200,:].copy()
    data_test = data.iloc[19200:,:].copy()
    data_train_valid.dropna(inplace=True)
    data_test.dropna(inplace=True)

    data_valid, data_train = train_test_split(data_train_valid, test_size=0.8, shuffle=False)
    return data_train.values, data_valid.values, data_test.values

def add_noise_to_rainfall(data_rainfall, hours_of_history, hours_to_predict, p_min, p_max):
    rainfall_and_noise = []
    for i in range(data_rainfall.shape[0]):
        for j in range(hours_to_predict):
            future_hour = hours_of_history + j
            # Denormalize the rainfall data
            denormalized_rainfall = data_rainfall[i, future_hour, 0] * (p_max - p_min) + p_min
            category = categorize_rainfall(denormalized_rainfall)
            if category is None:
                print(f"Sample {i}, Hour {future_hour}, Rainfall {denormalized_rainfall}: Category is None")
            if category:
                forecast_file = f"ft{j+1}.csv"
                dist, params = get_distribution_params(forecast_file, category)
                if dist and params:
                    noise = sample_noise_from_distributions(dist, params)
                    # Store the original rainfall, noise values, category, and forecast file
                    rainfall_and_noise.append({'sample': i, 'future_hour': future_hour, 'rainfall': denormalized_rainfall, 'noise': noise, 'category': category, 'forecast_file': forecast_file})
                    # Add noise
                    # Adjust the forecasted rainfall by subtracting noise to simulate forecasted data
                    denormalized_rainfall = max(denormalized_rainfall - noise, 0)  # Ensure non-negative values
                    # Normalize it back
                    data_rainfall[i, future_hour, 0] = (denormalized_rainfall - p_min) / (p_max - p_min)
    return data_rainfall, rainfall_and_noise

def prepare_data_with_noise(hours_of_history, hours_to_predict, parameters_included):
    """Prepare data for training and evaluation with added noise to future rainfall."""
    data = pd.read_csv('precip_streamflow_data.csv')

    scaler = MinMaxScaler()
    scaler.fit(data.iloc[:19200,:])
    
    p_max = np.max(data.iloc[:19200,0])
    p_min = np.min(data.iloc[:19200,0])
    
    q_max = np.max(data.iloc[:19200,1])
    q_min = np.min(data.iloc[:19200,1])
    
    data_scaled = scaler.transform(data)
    
    data_sequence = series_to_supervised(data_scaled, hours_of_history, hours_to_predict)
    data_train, data_valid, data_test = train_valid_test_split(data_sequence, hours_of_history, hours_to_predict, parameters_included)

    train_x_rainfall = data_train[:,0::2].reshape(-1, hours_of_history+hours_to_predict, 1)
    train_discharge = data_train[:,1::2].reshape(-1, hours_of_history+hours_to_predict, 1)
    train_x_discharge = train_discharge[:,:hours_of_history,:]
    train_y = train_discharge[:,hours_of_history:,:]
    

    # Add noise to training data
    train_x_rainfall, rainfall_and_noise_train = add_noise_to_rainfall(train_x_rainfall, hours_of_history, hours_to_predict, p_min, p_max)
    

    # Save the rainfall and noise values to a CSV file
    rainfall_and_noise_train_df = pd.DataFrame(rainfall_and_noise_train)
    



    valid_x_rainfall = data_valid[:,0::2].reshape(-1, hours_of_history+hours_to_predict, 1)
    valid_discharge = data_valid[:,1::2].reshape(-1, hours_of_history+hours_to_predict, 1)
    valid_x_discharge = valid_discharge[:,:hours_of_history,:]
    valid_y = valid_discharge[:,hours_of_history:,:]

    test_x_rainfall = data_test[:,0::2].reshape(-1, hours_of_history+hours_to_predict, 1)
    test_discharge = data_test[:,1::2].reshape(-1, hours_of_history+hours_to_predict, 1)
    test_x_discharge = test_discharge[:,:hours_of_history,:]
    test_y = test_discharge[:,hours_of_history:,:]

    # Add noise to test data
    test_x_rainfall, rainfall_and_noise_test = add_noise_to_rainfall(test_x_rainfall, hours_of_history, hours_to_predict, p_min, p_max)

    # Save the rainfall and noise values to a CSV file
    rainfall_and_noise_test_df = pd.DataFrame(rainfall_and_noise_test)
    
    
    # --------------------------------------
    # rescale targets back to physical units
    # --------------------------------------
    train_y = train_y * (q_max - q_min) + q_min
    valid_y = valid_y * (q_max - q_min) + q_min
    test_y  = test_y  * (q_max - q_min) + q_min
    
    
    return [train_x_discharge, train_x_rainfall], train_y, [valid_x_discharge, valid_x_rainfall], valid_y, [test_x_discharge, test_x_rainfall], test_y, q_max, q_min




def negative_log_likelihood(y_true, y_pred):
    mu, sigma = tf.split(y_pred, num_or_size_splits=2, axis=-1)

    sigma = tf.nn.softplus(sigma) + 1e-6
    sigma = tf.clip_by_value(sigma, 1e-6, 2.0)

    mu = tf.clip_by_value(mu, -6, 6)

    epsilon = 1e-6
    y_true = tf.clip_by_value(y_true, epsilon, np.inf)

    dist = tfp.distributions.LogNormal(loc=mu, scale=sigma)
    log_likelihood = dist.log_prob(y_true)
    nll = -tf.reduce_mean(log_likelihood)

    return nll



def EDLSTM(hours_of_history, hours_to_predict, parameters_included):
    """Define the EDLSTM model architecture."""
    input_1 = Input(shape=(hours_of_history, 1), name='LSTM1_input')
    LSTM1 = LSTM(256, return_sequences=False)(input_1)

    input_2 = Input(shape=((hours_of_history + hours_to_predict), 1), name='LSTM2_input')
    LSTM2 = LSTM(256, return_sequences=False, dropout=0.4)(input_2)

    x = concatenate([LSTM1, LSTM2])
    x = RepeatVector(hours_to_predict)(x)

    x = LSTM(512, return_sequences=True)(x)

    dim_dense = [512, 256, 128, 64]
    for dim in dim_dense:
        x = TimeDistributed(Dense(dim, activation='relu'))(x)
        x = TimeDistributed(Dropout(0.2))(x)

    log_mean_output = TimeDistributed(Dense(1))(x)
    log_std_output = TimeDistributed(Dense(1))(x)
    # log_std_output = tf.math.softplus(log_std_output)
    
    output = concatenate([log_mean_output, log_std_output], axis=-1)
    model = Model(inputs=[input_1, input_2], outputs=output)
    
    return model



def nse(y_true, y_pred):
    """Calculate Nash-Sutcliffe Efficiency (NSE)."""
    return 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)
  
def kge(y_true, y_pred):
    """Calculate Kling-Gupta Efficiency (KGE)."""
    kge_r = np.corrcoef(y_true, y_pred)[1][0]
    kge_a = np.std(y_pred) / np.std(y_true)
    kge_b = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((kge_r - 1)**2 + (kge_a - 1)**2 + (kge_b - 1)**2)
                                         
def main():
    """Main function to load data, train the model, and evaluate performance."""
    hours_to_predict = 12
    hours_of_history = 60
    parameters_included = 2

    batch_size = 64
    lr = 0.0001
    epochs = 300

    dest_path = 'your_path'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    test_name = os.path.join(dest_path, 'model')

    # Load data
    x_train, y_train, x_valid, y_valid, x_test, y_test, q_max, q_min = prepare_data_with_noise(hours_of_history, hours_to_predict, parameters_included)
    model1 = EDLSTM(hours_of_history, hours_to_predict, parameters_included)
    
    

    # avoid zero streamflows for log-normal (for more control)
    epsilon = 1e-6
    y_train = np.maximum(y_train, epsilon)
    y_valid = np.maximum(y_valid, epsilon)
    y_test  = np.maximum(y_test, epsilon)
    

    # Compile settings
    earlystoping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(test_name+'model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    optimizer = Adam(learning_rate=lr)
    
    model1.compile(optimizer=optimizer, loss=negative_log_likelihood)

    # Train model
    history = model1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                         validation_data=(x_valid, y_valid), callbacks=[earlystoping, checkpoint], verbose=1)

    # Save training loss
    loss_train = history.history['loss']
    loss_valid = history.history['val_loss']
    loss_train = pd.DataFrame({'TrainLoss': loss_train})
    loss_valid = pd.DataFrame({'TestLoss': loss_valid})
    LossEpoches = pd.concat([loss_train, loss_valid], axis=1)
    LossEpoches.to_csv(test_name+'loss.csv', index=True)

    # Final Test Review
    model1.load_weights(test_name+'model.h5')

    # Predict and save predictions
    y_pred_scaled = model1.predict(x_test)
    y_pred_mean_scaled = y_pred_scaled[:, :, 0]
    y_pred_std_scaled = y_pred_scaled[:, :, 1]
  
    new_column_names = ["mean_time_%s" % i for i in range(y_pred_mean_scaled.shape[1])]
    new_column_names_1 = ["std_time_%s" % i for i in range(y_pred_mean_scaled.shape[1])]
    df = pd.DataFrame(y_pred_mean_scaled, columns=new_column_names)
    df1= pd.DataFrame(y_pred_std_scaled, columns=new_column_names_1)
    combined_df = pd.concat([df, df1], axis=1)
    combined_df.to_csv(test_name+'predictions_scaled.csv', index=False)

    print("Prediction results saved.")
    
    import pdb;pdb.set_trace()    

if __name__ == "__main__":
    main()
