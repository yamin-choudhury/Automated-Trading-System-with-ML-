# This is a sample Python script.
import schedule
import time
import os


def hello_world():
    print("Hello World")

schedule.every(10).seconds.do(hello_world)

while 1:
    schedule.run_pending()
    time.sleep(1)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, \
    TimeDistributed, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Define the Transformer model
def TransformerForex(seq_len, d_model, num_heads, dff, dropout_rate):
    inputs = Input(shape=(seq_len, 1))

    # encode sequence using positional embeddings
    positions = tf.range(seq_len, dtype=tf.float32)
    positional_embeddings = tf.keras.layers.Embedding(seq_len, d_model)(positions)
    encoded = inputs + positional_embeddings

    # multi-head attention layer
    attention_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)([encoded, encoded])
    attention_out = Dropout(dropout_rate)(attention_out)
    attention_out = LayerNormalization(epsilon=1e-6)(inputs + attention_out)

    # feed-forward layer
    ffn = TimeDistributed(Dense(dff, activation='relu'))(attention_out)
    ffn = TimeDistributed(Dropout(dropout_rate))(ffn)
    ffn = TimeDistributed(Dense(1))(ffn)
    outputs = Reshape((seq_len,))(ffn)

    # create model
    model = Model(inputs=inputs, outputs=outputs)

    return model


# Load and preprocess the data
data = pd.read_csv('forex_data.csv')
scaler = MinMaxScaler()
data['Price'] = scaler.fit_transform(data[['Price']])
seq_len = 30

# Create training, validation, and test datasets
train_data = data.iloc[:800]
val_data = data.iloc[800:1000]
test_data = data.iloc[1000:]

# Create input and output sequences for training dataset
train_X = []
train_y = []
for i in range(seq_len, len(train_data)):
    train_X.append(train_data.iloc[i-seq_len:i]['Price'].values)
    train_y.append(train_data.iloc[i]['Price'])
train_X = np.array(train_X)
train_y = np.array(train_y)

# Create input and output sequences for validation dataset
val_X = []
val_y = []
for i in range(seq_len, len(val_data)):
    val_X.append(val_data.iloc[i-seq_len:i]['Price'].values)
    val_y.append(val_data.iloc[i]['Price'])
val_X = np.array(val_X)
val_y = np.array(val_y)

# Create input and output sequences for test dataset
test_X = []
test_y = []
for i in range(seq_len, len(test_data)):
    test_X.append(test_data.iloc[i-seq_len:i]['Price'].values)
    test_y.append(test_data.iloc[i]['Price'])
test_X = np.array(test_X)
test_y = np.array(test_y)

# Define model parameters
d_model = 32
num_heads = 2
dff = 64
dropout_rate = 0.1

# Create and compile the model
model = TransformerForex(seq_len, d_model, num_heads, dff, dropout_rate)
optimizer = Adam(lr=1e-3)
model.compile(loss='mse', optimizer=optimizer)

# Train the model
model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=100, batch_size=32)

# Predict the next 10 future iterations
# Predict the next 10 future iterations
last_sequence = test_X[-1]
forecast = []
for i in range(10):
    next_pred = model.predict(last_sequence.reshape(1, seq_len, 1))
    forecast.append(next_pred[0])
    last_sequence = np.append(last_sequence[1:], next_pred)

# Inverse transform the forecast and actual values
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
actuals = scaler.inverse_transform(test_y.reshape(-1, 1))

# Print the forecast and actual values
for i in range(len(forecast)):
    print(f"Predicted price for time step {i+1}: {forecast[i][0]:.5f}")
print(f"\nActual prices:\n{actuals.flatten()[-10:]}\n")
