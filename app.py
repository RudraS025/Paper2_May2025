from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load pipeline and preprocessors
pipeline = joblib.load('rf_pipeline.save')
imputer = joblib.load('imputer.save')
log_transformer = joblib.load('log_transformer.save')
important_features = joblib.load('selected_features.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    # Ensure columns are in the right order and only the important features
    df = df[important_features]
    # Impute and log-transform
    X_imputed = pd.DataFrame(imputer.transform(df), columns=important_features)
    X_log = pd.DataFrame(log_transformer.transform(X_imputed), columns=important_features)
    y_pred = pipeline.predict(X_log)
    return jsonify({'predictions': y_pred.tolist()})

@app.route('/form', methods=['GET', 'POST'])
def form():
    results = None
    if request.method == 'POST':
        try:
            # Collect up to 5 rows of input
            rows = []
            for i in range(5):
                date_val = request.form.get(f'date_{i}', '').strip()
                row = []
                empty = True
                for feat in important_features:
                    val = request.form.get(f'{feat}_{i}', '').strip()
                    if val != '':
                        empty = False
                    row.append(val)
                if not empty and date_val:
                    rows.append({'date': date_val, 'values': row})
            if not rows:
                raise Exception('Please provide at least one row of input with date and all variables.')
            # Prepare DataFrame for prediction
            X = []
            dates = []
            for row in rows:
                # Convert Excel serial date to string if needed
                date_val = row['date']
                try:
                    # Try to convert if it's a number (Excel serial date)
                    if date_val and str(date_val).replace('.', '', 1).isdigit():
                        # Excel's origin is 1899-12-30
                        date_val = str(pd.to_datetime(float(date_val), origin='1899-12-30', unit='D').date())
                except Exception:
                    pass
                X.append([float(v) for v in row['values']])
                dates.append(date_val)
            arr = np.array(X)
            arr_imputed = imputer.transform(arr)
            arr_log = log_transformer.transform(arr_imputed)
            preds = pipeline.predict(arr_log)
            results = [ {'date': dates[i], 'forecast': round(float(preds[i]), 2)} for i in range(len(preds)) ]
        except Exception as e:
            results = [{'date': '', 'forecast': f"Error: {e}"}]
    return render_template('form.html', results=results, features=important_features)

if __name__ == '__main__':
    app.run(debug=True)