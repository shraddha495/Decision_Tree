import streamlit as st
import pickle
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="wide"
)

# Custom CSS Styling inside app.py
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        height: 3em;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    h1, h2 {
        color: #1f3bb3;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained Decision Tree model
@st.cache_resource
def load_model():
    with open("decision.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("💳 Loan Approval Prediction App")
st.markdown("Enter the applicant's details below to check loan eligibility using your trained Decision Tree model (`v1.6.1`)[cite: 1].")
st.markdown("---")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographic & Personal Info")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    
    # Categorical columns in category form
    education = st.selectbox("Education Status", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
    
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700, step=1)

with col2:
    st.subheader("💰 Financial & Asset Details")
    income_annum = st.number_input("Annual Income (₹)", min_value=0, value=600000, step=10000)
    loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0, value=2000000, step=10000)
    residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=1500000, step=10000)
    commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=0, step=10000)
    luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=200000, step=10000)
    bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=500000, step=10000)

st.markdown("---")

# Prediction Trigger
if st.button("Predict Loan Status"):
    # Construct DataFrame with exact feature names matching the model
    input_data = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=[
        'no_of_dependents', 'education', 'self_employed', 'income_annum', 
        'loan_amount', 'cibil_score', 'residential_assets_value', 
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ])
    
    # NOTE: If your training pipeline used Label Encoding/Ordinal Encoding for 
    # 'education' or 'self_employed', map them here before running model.predict().
    # Example:
    # input_data['education'] = input_data['education'].map({'Graduate': 0, 'Not Graduate': 1})
    # input_data['self_employed'] = input_data['self_employed'].map({'No': 0, 'Yes': 1})

    try:
        prediction = model.predict(input_data)
        
        st.subheader("📋 Prediction Result")
        if prediction[0] == 1 or str(prediction[0]).lower() in ['approved', 'y', '1']:
            st.success("🎉 Congratulations! The Loan application is **APPROVED**.")
        else:
            st.error("❌ Sorry, the Loan application is **REJECTED**.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Tip: Verify if your model expects categorical text strings or encoded numeric labels.")
