from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
"""
to run : py -m metrics.Metric4.ml.neural_network
"""

data = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric4/ml/processed_data.csv')
data = data.sample(frac=1).reset_index(drop=True)

X = data.drop(columns=['symbol', 'date', 'to_buy'])
y = data['to_buy']

"""X = data[['divpayout_ratio', 'std_10d', 'RnD_expenses', 'income_lag_days',
       'marketCap', 'total_debt', 'curr_est_eps', 'var_price_to_minmax',
       'maxPercen_4M', 'debtToPrice', 'markRevRAtio', 'eps_growth', 'pe',
       'peg', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale', 'var_sma50D_100D',
       'relative_std', 'deriv_1w', 'deriv_1m', 'deriv_3m', '1W_return',
       '1M_return', '7M_return', '1y_return', '2M_1M_growth', '3M_2M_growth',
       '4M_3M_growth', '1Y_6M_growth']]"""

X= data[['price', 'marketCap', 'balance_lag_days', 'max_minus_min',
        'markRevRatio', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP',
       'EVRevenues', 'fwdPriceTosale_diff', 'deriv_4m', '3M_return',
       'sma10_yoy_growth', 'peg_sector_relative', '3M_return_sector_relative',
       '6M_return_sector_relative', '2M_return_sector_relative']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

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
    factor=0.8,         # Multiply learning rate by 0.5 when plateauing
    patience=30,         # Number of epochs with no improvement after which learning rate will be reduced
    min_delta=1e-4,     # Minimum change in monitored value to qualify as an improvement
    min_lr=1e-6,        # Lower bound on the learning rate
    verbose=1           # Print message when reducing learning rate
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(16),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=350, batch_size=32, validation_split=0.08, callbacks=[reduce_lr],verbose=1)

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