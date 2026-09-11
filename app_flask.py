from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("decision.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = pd.DataFrame([[
            data['no_of_dependents'],
            data['education'],
            data['self_employed'],
            data['income_annum'],
            data['loan_amount'],
            data['cibil_score'],
            data['residential_assets_value'],
            data['commercial_assets_value'],
            data['luxury_assets_value'],
            data['bank_asset_value']
        ]], columns=[
            'no_of_dependents', 'education', 'self_employed', 'income_annum', 
            'loan_amount', 'cibil_score', 'residential_assets_value', 
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
        ])
        
        prediction = model.predict(input_data)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
