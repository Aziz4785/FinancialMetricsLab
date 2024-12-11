from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
"""
to run : py -m metrics.Metric8.ml.neural_network
"""

data = pd.read_csv('metrics/Metric8/ml/processed_data.csv')
data = data.sample(frac=1).reset_index(drop=True)

#X = data.drop(columns=['symbol', 'date', 'to_buy'])
X = data.drop(columns=['symbol', 'date', 'to_buy','sector','max_in_8M','sma_10d_7months_ago','var_sma10D_50D','total_debt',
                     'max_in_1M','min_in_8M','sma_50w','min_in_1M','min_in_4M','max_in_2W','var_sma50D','5M_return',
                     'sma_10d_4months_ago','vwap','sma_10d_1months_ago','var_sma50D_100D','sma_50d_to_sma_200d_ratio',
                     'sma_10d_2weeks_ago','income_lag_days','price_to_sma_10d_ratio','sma_50d_to_sma_100d_ratio','price',
                     'max_in_4M','sma_10d_1weeks_ago','sma_200d','sma10_yoy_growth','price_to_sma_200d_ratio','std_10d',
                     'min_in_2W','sma_100d','1W_return','var_sma100D','revenues','EVRevenues','pe','sma_10d_11months_ago',
                     'curr_est_rev','sma_10d','sma_50d','eps_growth','death_cross','sma_10d_5months_ago','var_sma10D_100D',
                     'sma_10d_3months_ago','open_price','sma_10d_2months_ago','EV','var_sma10D_200D'])
y = data['to_buy']


# unique_features = list(set(all_possible_features))
# print(unique_features)

# X= data[unique_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("length of X train : ",len(X_train))
print("length of X_test : ",len(X_test))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(np.isnan(X_train_scaled).sum(), np.isnan(y_train).sum())  # Should be 0
print(np.isinf(X_train_scaled).sum(), np.isinf(y_train).sum())  # Should be 0

print(X.columns)

custom_learning_rate = 0.0008
optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.75,         # Multiply learning rate by 0.5 when plateauing
    patience=30,         # Number of epochs with no improvement after which learning rate will be reduced
    min_delta=1e-4,     # Minimum change in monitored value to qualify as an improvement
    min_lr=1e-7,        # Lower bound on the learning rate
    verbose=1           # Print message when reducing learning rate
)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],),
#                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],),
                          kernel_regularizer=tf.keras.regularizers.l2(0.008)),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.008)),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],),
#                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(16, activation='relu',
#                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, validation_split=0.12, callbacks=[reduce_lr],verbose=1)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)


# Plot the training history
plt.figure(figsize=(14, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()