from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load pipeline
pipeline = joblib.load('rf_pipeline.save')

# List of feature names in the correct order
FEATURES = [
    "Mixed wastepaper FOB (USD/ton)",
    "US ISM, Manufacturing, Suppliers Delivery Index (thousand units)",
    "Paperboard mills  (NAICS = 32213); n.s.a. IP",
    "Paperboard container  (NAICS = 32221); n.s.a. IP",
    "US Recovered Paper Exports ('000 tons)",
    "US Kraft Paper Imports ('000 tons)",
    "US Kraft Paper Exports (thousand tons)",
    "Waste management SA (thousand units) - people",
    "Waste management NSA  (thousand units) - people",
    "waste collection sa  (thousand units) - people",
    "waste collection nsa  (thousand units) - people",
    "solid waste collection  sa  (thousand units) - people",
    "solid waste collection nsa  (thousand units) - people",
    "Solid waste landfill SA  (thousand units) - people",
    "Solid waste landfill NSA  (thousand units) - people",
    "materials recovery  SA  (thousand units)",
    "materials recovery NSA  (thousand units)",
    "Retail and food services sales, total",
    "Motor vehicle and parts dealers",
    "Nonstore retailers",
    "Food and beverage stores",
    "General merchandise stores",
    "Food services and drinking places",
    "Building mat. and garden equip. and supplies dealers",
    "Gasoline stations",
    "Health and personal care stores",
    "Clothing and clothing access. stores",
    "Furniture, home furn, electronics, and appliance stores",
    "Miscellaneous stores retailers",
    "Sporting goods, hobby, musical instrument, and book stores"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    y_pred = pipeline.predict(df)
    return jsonify({'predictions': y_pred.tolist()})

@app.route('/form', methods=['GET', 'POST'])
def form():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form.get(feat, 0)) for feat in FEATURES]
            arr = np.array([input_data])
            pred = pipeline.predict(arr)[0]
            prediction = round(float(pred), 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)