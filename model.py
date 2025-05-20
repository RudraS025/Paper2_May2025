import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the imputed data
df = pd.read_excel('Paper_May25_imputed.xlsx')

DATE_COL = 'Months'
TARGET_COL = 'OCC FOB (USD/ton)'

# Separate features and target
X = df.drop([DATE_COL, TARGET_COL], axis=1)
y = df[TARGET_COL]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from tensorflow import keras
from tensorflow.keras import layers

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=1)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict on test set
y_pred = model.predict(X_test_scaled).flatten()

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title('Actual vs Predicted OCC FOB (USD/ton)')
plt.xlabel('Sample')
plt.ylabel('OCC FOB (USD/ton)')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()
print("Actual vs Predicted plot saved as actual_vs_predicted.png")

# Save actual and predicted values to Excel
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_excel('OCC_FOB_predictions.xlsx', index=False)
print('Predictions saved to OCC_FOB_predictions.xlsx')

import joblib

# Save scaler
joblib.dump(scaler, 'scaler.save')

# Save model
model.save('my_model.h5')

# --- Inference on New Data ---

# Load new data (replace 'NewData.xlsx' with your actual file name)
new_df = pd.read_excel('NewData.xlsx')

# Drop date and target columns if present
X_new = new_df.drop(['Months', 'OCC FOB (USD/ton)'], axis=1, errors='ignore')

# Scale new data using the same scaler as training
X_new_scaled = scaler.transform(X_new)

# Predict using the trained model
y_new_pred = model.predict(X_new_scaled).flatten()

# Save predictions to Excel
new_df['Predicted_OCC_FOB'] = y_new_pred
new_df.to_excel('NewData_with_predictions.xlsx', index=False)
print('Predictions for new data saved to NewData_with_predictions.xlsx')