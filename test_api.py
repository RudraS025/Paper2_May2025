import requests
import pandas as pd

# Prepare a sample row with the correct feature names and order (excluding 'OCC FOB (USD/ton)')
data = [
    {
        "Mixed wastepaper FOB (USD/ton)": 180,
        "US ISM, Manufacturing, Suppliers Delivery Index (thousand units)": 290,
        "Paperboard mills  (NAICS = 32213); n.s.a. IP": 220,
        "Paperboard container  (NAICS = 32221); n.s.a. IP": 210,
        "US Recovered Paper Exports ('000 tons)": 320,
        "US Kraft Paper Imports ('000 tons)": 310,
        "US Kraft Paper Exports (thousand tons)": 300,
        "Waste management SA (thousand units) - people": 360,
        "Waste management NSA  (thousand units) - people": 350,
        "waste collection sa  (thousand units) - people": 340,
        "waste collection nsa  (thousand units) - people": 330,
        "solid waste collection  sa  (thousand units) - people": 240,
        "solid waste collection nsa  (thousand units) - people": 250,
        "Solid waste landfill SA  (thousand units) - people": 270,
        "Solid waste landfill NSA  (thousand units) - people": 260,
        "materials recovery  SA  (thousand units)": 60,
        "materials recovery NSA  (thousand units)": 65,
        "Retail and food services sales, total": 230,
        "Motor vehicle and parts dealers": 160,
        "Nonstore retailers": 170,
        "Food and beverage stores": 130,
        "General merchandise stores": 150,
        "Food services and drinking places": 140,
        "Building mat. and garden equip. and supplies dealers": 100,
        "Gasoline stations": 90,
        "Health and personal care stores": 80,
        "Clothing and clothing access. stores": 120,
        "Furniture, home furn, electronics, and appliance stores": 110,
        "Miscellaneous stores retailers": 70,
        "Sporting goods, hobby, musical instrument, and book stores": 280
    }
]

response = requests.post("http://127.0.0.1:5000/predict", json=data)
print(response.json())