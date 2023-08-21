import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np

def normalise(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val

def standardise(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val, mean_val, std_val
    
def simple_rf(X_train, y_train):
    #initialize
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    #train
    model = clf.fit(X_train, y_train)
    return model

def simple_nn(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def one_hot_labels(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))
    labels_encoded = to_categorical(labels, num_classes=num_classes)
    return labels_encoded
    
def regression_nn(input_dim=768):
    model = Sequential()
    #input Layer
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    #hidden Layers
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    #output layer
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def regression_model2(input_dim):
  model = Sequential([
    #first hidden layer
    Dense(512, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.5), #Dropout for regularization

    #second hidden layer
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), #Dropout for regularization

    #third hidden layer
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), # Dropout for regularization

    #output layer
    Dense(1, activation='linear')])
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  return model
  
def regression_model_to_opt(input_dim, hidden_units, dropout_rate):
    model = Sequential([
        Dense(hidden_units[0], activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate[0]),

        Dense(hidden_units[1], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate[1]),

        Dense(hidden_units[2], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate[2]),

        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def objective(trial, X_train, X_val, y_train, y_val):
    # Define hyperparameter search space using trial object
    hidden_units = [
        trial.suggest_int("hidden_units_1", 128, 1024, step=128),
        trial.suggest_int("hidden_units_2", 64, 512, step=64),
        trial.suggest_int("hidden_units_3", 32, 256, step=32)]
    dropout_rate = [
        trial.suggest_float("dropout_rate_1", 0.1, 0.7, step=0.1),
        trial.suggest_float("dropout_rate_2", 0.1, 0.7, step=0.1),
        trial.suggest_float("dropout_rate_3", 0.1, 0.7, step=0.1)]

    model = regression_model2(X_train.shape[1], hidden_units, dropout_rate)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=512, verbose=0)
    val_loss = history.history["val_loss"][-1]
    return val_loss
    
def regression_model_opt(input_dim):
    #hyperparameters from Optuna trial 31
    hu1 = 256
    hu2 = 448
    hu3 = 32
    dr1 = 0.2
    dr2 = 0.1
    dr3 = 0.4

    model = Sequential([
        Dense(hu1, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dr1),

        Dense(hu2, activation='relu'),
        BatchNormalization(),
        Dropout(dr2),

        Dense(hu3, activation='relu'),
        BatchNormalization(),
        Dropout(dr3),

        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

def simpleCNN(input_dim=(160, 1536, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    
    #compile the model for regression
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
  
def regression_report(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    residuals = y_true - y_pred
    correlation_coefficient, _ = pearsonr(y_true, y_pred)

    report = {
        'R^2 Score': r2,
        'Mean Absolute Error (MAE)': mae,
        'Root Mean Squared Error (RMSE)': rmse,
        'Pearson Correlation coefficient': correlation_coefficient,
        'Residuals': residuals
    }

    return report
    
