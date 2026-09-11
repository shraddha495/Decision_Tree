import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample training data matching your 10 features
data = {
    'no_of_dependents': [2, 0, 1, 3],
    'education': [0, 1, 0, 1], # 0: Graduate, 1: Not Graduate
    'self_employed': [0, 1, 0, 0], # 0: No, 1: Yes
    'income_annum': [5000000, 2000000, 4500000, 1500000],
    'loan_amount': [15000000, 5000000, 12000000, 4000000],
    'cibil_score': [750, 600, 800, 550],
    'residential_assets_value': [4000000, 1000000, 3000000, 800000],
    'commercial_assets_value': [2000000, 0, 1500000, 0],
    'luxury_assets_value': [5000000, 1000000, 4000000, 500000],
    'bank_asset_value': [2000000, 500000, 1800000, 200000],
    'loan_status': [1, 0, 1, 0] # 1: Approved, 0: Rejected
}

df = pd.DataFrame(data)

X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Train the model properly
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the fitted model
with open('decision.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Successfully generated a valid, fitted 'decision.pkl' file!")
