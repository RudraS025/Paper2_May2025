import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load the revised data
print('Loading data...')
df = pd.read_excel('Paper_Revised_May25.xlsx')

# 2. Separate columns
DATE_COL = df.columns[0]  # First column is Month
TARGET_COL = df.columns[1]  # Second column is target
FEATURE_COLS = df.columns[2:]  # All other columns are features

# 3. Impute missing values (except Month and target)
print('Imputing missing values...')
features = df[FEATURE_COLS]
imputer = IterativeImputer(random_state=42)
features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=FEATURE_COLS)

# 4. Log transform features (to reduce skewness)
print('Applying log transformation...')
log_transformer = FunctionTransformer(np.log1p)
features_log = pd.DataFrame(log_transformer.fit_transform(features_imputed), columns=FEATURE_COLS)

# 5. Feature selection: Remove features with very low correlation to target
print('Selecting features based on correlation...')
target = df[TARGET_COL]
correlations = features_log.corrwith(target)
selected_features = correlations[correlations.abs() > 0.1].index.tolist()  # Keep features with |corr| > 0.1
print(f'Selected features (correlation > 0.1): {selected_features}')

# Only keep selected features for all further steps
features_selected = features[selected_features]
features_imputed_selected = pd.DataFrame(imputer.fit_transform(features_selected), columns=selected_features)
features_log_selected = pd.DataFrame(log_transformer.fit_transform(features_imputed_selected), columns=selected_features)

X = features_log_selected
y = target

# 6. Further feature selection using RandomForest feature importances
print('Further feature selection using RandomForest...')
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_temp.fit(X, y)
importances = pd.Series(rf_temp.feature_importances_, index=X.columns)
important_features = importances[importances > importances.mean() * 0.5].index.tolist()  # Keep features above half mean importance
print(f'Final selected features: {important_features}')

# === REFIT IMPUTER AND LOG TRANSFORMER ON FINAL FEATURES ===
features_final = features[important_features]
imputer_final = IterativeImputer(random_state=42)
features_imputed_final = pd.DataFrame(imputer_final.fit_transform(features_final), columns=important_features)
log_transformer_final = FunctionTransformer(np.log1p)
features_log_final = pd.DataFrame(log_transformer_final.fit_transform(features_imputed_final), columns=important_features)

X = features_log_final

# 7. Train/test split (use only important_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Build pipeline: scaling + RandomForest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 9. Train the pipeline
print('Training model...')
pipeline.fit(X_train, y_train)

# 10. Predict on test set
y_pred = pipeline.predict(X_test)

# 11. Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

# 12. Save actual and predicted values to Excel
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_excel('OCC_FOB_predictions.xlsx', index=False)
print('Predictions saved to OCC_FOB_predictions.xlsx')

# 13. Save the pipeline and feature list
joblib.dump(pipeline, 'rf_pipeline.save')
joblib.dump(imputer_final, 'imputer.save')
joblib.dump(log_transformer_final, 'log_transformer.save')
joblib.dump(important_features, 'selected_features.save')
print('Pipeline and preprocessors saved.')

# 14. --- Inference on New Data ---
try:
    new_df = pd.read_excel('Future_Data_Paper_Revised.xlsx')
    new_df.columns = [col.strip() for col in new_df.columns]
    print('Columns in new_df:', list(new_df.columns))
    print('Columns expected:', important_features)
    X_new = new_df[important_features].copy()
    # Impute and log-transform using the FINAL preprocessors
    X_new_imputed = pd.DataFrame(imputer_final.transform(X_new), columns=important_features)
    X_new_log = pd.DataFrame(log_transformer_final.transform(X_new_imputed), columns=important_features)
    y_new_pred = pipeline.predict(X_new_log)
    new_df['Predicted_OCC_FOB'] = y_new_pred
    new_df.to_excel('NewData_with_predictions.xlsx', index=False)
    print('Predictions for new data saved to NewData_with_predictions.xlsx')
except Exception as e:
    print(f'No new data for inference or error: {e}')