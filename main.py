# Step 1: Import libraries
import pandas as pd

# Step 2: Load data (show raw columns and first few rows)
file_path = 'Paper_May25.xlsx'
df_raw = pd.read_excel(file_path)
print('Raw data shape:', df_raw.shape)
print('Raw columns:', df_raw.columns.tolist())
print(df_raw.head(10))

# Print all column names so you can confirm them
print('Columns in your data:', df_raw.columns.tolist())

# Set your actual date and target column names here
DATE_COL = 'Months'  # Change if your date column is named differently
TARGET_COL = 'OCC FOB (USD/ton)'  # Change if your target column is named differently

# Convert all columns except date to numeric
for col in df_raw.columns:
    if col != DATE_COL:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# Impute missing values (excluding date)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_impute = df_raw.copy()
date_values = df_impute[DATE_COL]
df_to_impute = df_impute.drop(DATE_COL, axis=1)
imputer = IterativeImputer(random_state=0)
df_imputed_values = imputer.fit_transform(df_to_impute)
df_imputed = pd.DataFrame(df_imputed_values, columns=df_to_impute.columns)
df_imputed[DATE_COL] = date_values.values

print('Missing values after imputation:')
print(df_imputed.isnull().sum())
print(df_imputed.head())

# Save the imputed DataFrame to a new Excel file
imputed_file = 'Paper_May25_imputed.xlsx'
df_imputed.to_excel(imputed_file, index=False)
print(f'Imputed data saved to {imputed_file}')
