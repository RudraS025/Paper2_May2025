import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# Load the imputed data
df = pd.read_excel('Paper_May25_imputed.xlsx')

DATE_COL = 'Months'
TARGET_COL = 'OCC FOB (USD/ton)'

# Separate features and target
X = df.drop([DATE_COL, TARGET_COL], axis=1)
y = df[TARGET_COL]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline: scaling + RandomForest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Save actual and predicted values to Excel
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_excel('OCC_FOB_predictions.xlsx', index=False)
print('Predictions saved to OCC_FOB_predictions.xlsx')

# Save the pipeline
joblib.dump(pipeline, 'rf_pipeline.save')
print('Pipeline saved as rf_pipeline.save')

# --- Inference on New Data ---
# Load new data (replace 'NewData.xlsx' with your actual file name)
new_df = pd.read_excel('NewData.xlsx')

# Drop date and target columns if present
X_new = new_df.drop(['Months', 'OCC FOB (USD/ton)'], axis=1, errors='ignore')

# Predict using the trained pipeline
y_new_pred = pipeline.predict(X_new)

# Save predictions to Excel
new_df['Predicted_OCC_FOB'] = y_new_pred
new_df.to_excel('NewData_with_predictions.xlsx', index=False)
print('Predictions for new data saved to NewData_with_predictions.xlsx')