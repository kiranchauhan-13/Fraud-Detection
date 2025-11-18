import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import streamlit as st # type: ignore
warnings.filterwarnings('ignore')   
sns.set(style="whitegrid")

# Example input data
# Suppose this is the input you want to pass
input_data = pd.DataFrame([{
    'oldbalanceDest': 10000,
    'newbalanceDest': 9500,
    'amount': 500,
    'type': 'CASH_OUT',
    # Add missing fields here
    'balanceDiff0rig': 100,
    'oldbalanceOrg': 11000
}])

# If a global `dt` dataframe already exists in the notebook, keep it.
# Otherwise, use the provided input_data as `dt`.
if 'dt' not in globals():
    dt = input_data.copy()
model = joblib.load('fraud_detection_model.pkl')    

st.title("Fraud Detection App")

st.markdown("please input the transaction details below:    ")

st.divider()


# Use the existing fitted pipeline instance if available, otherwise try to load it from disk
if 'Pipeline' in globals():
	model = Pipeline # type: ignore
else:
	# Load saved pipeline; avoid re-importing joblib if it's already present
	if 'joblib' not in globals():
		import joblib
	model = joblib.load('fraud_detection_model.pkl')

# Safely access the preprocessor step and print the numeric transformer column indices
if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
	pre = model.named_steps['preprocessor']
	print(pre.transformers_[0][2])
else:
	raise AttributeError("The model does not contain a 'preprocessor' step.")


transaction_type = st.selectbox("Transaction Type", options=['CASH_IN','DEPOSIT',   'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
amount = st.number_input("Transaction Amount", min_value=0.0, step=1000.0)
oldbalanceOrig = st.number_input("Old Balance (Sender)", min_value=0.0, step=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, step=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, step=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, step=0.0)

if st.button("Predict Fraud"):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrig': [oldbalanceOrig],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'balanceDiffOrig': [oldbalanceOrig - newbalanceOrig],
        'balanceDiffDest': [newbalanceDest - oldbalanceDest]
    })



    prediction = model.predict(input_data)

    st.subheader(f"Prediction: {int(prediction[0])}")

    if prediction[0] == 1:
        st.error("The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("The transaction is predicted to be LEGITIMATE.")

