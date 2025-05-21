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
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form.get(feat, 0)) for feat in important_features]
            arr = np.array([input_data])
            # Impute and log-transform
            arr_imputed = imputer.transform(arr)
            arr_log = log_transformer.transform(arr_imputed)
            pred = pipeline.predict(arr_log)[0]
            prediction = round(float(pred), 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('form.html', prediction=prediction, features=important_features)

if __name__ == '__main__':
    app.run(debug=True)