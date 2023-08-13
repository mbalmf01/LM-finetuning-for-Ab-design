import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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