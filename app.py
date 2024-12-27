import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define mappings for input
education_mapping = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}
marriage_mapping = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}

# App title and description
st.set_page_config(page_title="Credit Default Prediction", page_icon="💳")
st.title("💳 Credit Card Default Prediction")
st.markdown("""
### 📋 Overview
Our app helps predict the default of credit card payments, using a machine learning model. 🔍 It generates accurate predictions along with their confidence scores based on clients' financial and demographic details. 📊 Highlighting whether the client would likely or unlikely default against the account, it always provides actionable insights. 💡 Take proactive steps using personalized suggestions for improving their financial stability. 📈 Try it now, and make informed decisions related to your credit risk!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("""
Use this sidebar to navigate through the app.
- **Prediction**: Make a prediction.
- **About**: Learn more about this app.
""")
page = st.sidebar.radio("Go to", ["Prediction", "About"])

if page == "Prediction":
    # Input Form
    st.subheader("📊 Client Details")
    st.markdown("Provide the client’s financial and personal information below:")

    with st.form(key='prediction_form'):
        # Credit limit
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0, step=1000, help="Enter the approved credit limit for the client.")
        
        # Demographics
        sex = st.selectbox("Sex", ["Male", "Female"], help="Select the client’s gender.")
        education = st.selectbox("Education Level", list(education_mapping.keys()), help="Select the highest level of education achieved.")
        marriage = st.selectbox("Marital Status", list(marriage_mapping.keys()), help="Select the client’s marital status.")
        age = st.number_input("Age", min_value=18, step=1, help="Enter the client’s age in years.")
        
        st.markdown("#### Payment History (Last 6 Months)")
        col1, col2 = st.columns(2)
        with col1:
            pay_status_sept = st.number_input("September (PAY_0)", step=1, help="Enter the repayment status for September.")
            pay_status_jul = st.number_input("July (PAY_3)", step=1, help="Enter the repayment status for July.")
            pay_status_may = st.number_input("May (PAY_5)", step=1, help="Enter the repayment status for May.")
        with col2:
            pay_status_aug = st.number_input("August (PAY_2)", step=1, help="Enter the repayment status for August.")
            pay_status_jun = st.number_input("June (PAY_4)", step=1, help="Enter the repayment status for June.")
            pay_status_apr = st.number_input("April (PAY_6)", step=1, help="Enter the repayment status for April.")
        
        st.markdown("#### Bill Amounts (Last 6 Months)")
        col1, col2 = st.columns(2)
        with col1:
            bill_amt_sept = st.number_input("September (BILL_AMT1)", step=100, help="Enter the bill amount for September.")
            bill_amt_jul = st.number_input("July (BILL_AMT3)", step=100, help="Enter the bill amount for July.")
            bill_amt_may = st.number_input("May (BILL_AMT5)", step=100, help="Enter the bill amount for May.")
        with col2:
            bill_amt_aug = st.number_input("August (BILL_AMT2)", step=100, help="Enter the bill amount for August.")
            bill_amt_jun = st.number_input("June (BILL_AMT4)", step=100, help="Enter the bill amount for June.")
            bill_amt_apr = st.number_input("April (BILL_AMT6)", step=100, help="Enter the bill amount for April.")
        
        st.markdown("#### Payment Amounts (Last 6 Months)")
        col1, col2 = st.columns(2)
        with col1:
            pay_amt_sept = st.number_input("September (PAY_AMT1)", step=100, help="Enter the payment amount for September.")
            pay_amt_jul = st.number_input("July (PAY_AMT3)", step=100, help="Enter the payment amount for July.")
            pay_amt_may = st.number_input("May (PAY_AMT5)", step=100, help="Enter the payment amount for May.")
        with col2:
            pay_amt_aug = st.number_input("August (PAY_AMT2)", step=100, help="Enter the payment amount for August.")
            pay_amt_jun = st.number_input("June (PAY_AMT4)", step=100, help="Enter the payment amount for June.")
            pay_amt_apr = st.number_input("April (PAY_AMT6)", step=100, help="Enter the payment amount for April.")
        
        # Buttons
        submit_button = st.form_submit_button(label="Predict")
        reset_button = st.form_submit_button(label="Reset")

    if submit_button:
        # Prepare input data
        user_input_data = pd.DataFrame({
            "LIMIT_BAL": [limit_bal],
            "SEX": [1 if sex == "Male" else 2],
            "EDUCATION": [education_mapping[education]],
            "MARRIAGE": [marriage_mapping[marriage]],
            "AGE": [age],
            "PAY_0": [pay_status_sept],
            "PAY_2": [pay_status_aug],
            "PAY_3": [pay_status_jul],
            "PAY_4": [pay_status_jun],
            "PAY_5": [pay_status_may],
            "PAY_6": [pay_status_apr],
            "BILL_AMT1": [bill_amt_sept],
            "BILL_AMT2": [bill_amt_aug],
            "BILL_AMT3": [bill_amt_jul],
            "BILL_AMT4": [bill_amt_jun],
            "BILL_AMT5": [bill_amt_may],
            "BILL_AMT6": [bill_amt_apr],
            "PAY_AMT1": [pay_amt_sept],
            "PAY_AMT2": [pay_amt_aug],
            "PAY_AMT3": [pay_amt_jul],
            "PAY_AMT4": [pay_amt_jun],
            "PAY_AMT5": [pay_amt_may],
            "PAY_AMT6": [pay_amt_apr]
        })

        # Get prediction probabilities (if your model supports it)
        prediction_proba = model.predict_proba(user_input_data)
        prediction = model.predict(user_input_data)

        # Display prediction result
        if prediction[0] == 1:
            st.error("🚨 **Result:** The client is likely to default on their credit card payment.")
            st.markdown("### 💡 Suggestions:")
            st.write("- Consider reducing the credit limit.")
            st.write("- Encourage the client to clear pending dues.")
        else:
            st.success("✅ **Result:** The client is unlikely to default on their credit card payment.")
            st.markdown("### 🎉 Good to Know:")
            st.write("- The client has a stable financial history.")
            st.write("- Encourage responsible payment practices.")

        # Plot the bar graph showing prediction probabilities
        proba_default = prediction_proba[0][1]
        proba_non_default = prediction_proba[0][0]

        fig, ax = plt.subplots()
        ax.bar(["Likely to Default", "Not Likely to Default"], [proba_default, proba_non_default], color=["red", "green"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

elif page == "About":
    st.subheader("About This App")
    st.markdown("""
    
    This application uses machine learning to predict whether a client is likely to default on their credit card payment based on their financial and demographic details. The model uses a variety of factors such as credit limit, payment history, bill amounts, and demographic information (age, sex, education, and marital status) to provide an accurate prediction.

    ### Features:
    - **Prediction**: Enter your client's information to get a prediction score on the probability that they will default on the credit card payment.
    - **Confidence Scores**: The app gives you a prediction along with a confidence level that guides you to know the percentage or degree of the outcome.
    - **User-Friendly Interface**: Just enter financial information, and you'll have immediate predictions.
    - **Actionable Insights**: Using the prediction, users can be offered some recommendations to enhance the financial health of the client or give appropriate suggestions.

    ### How it Works:
    The app employs a trained machine learning model that analyzes past credit data and makes predictions based on the inputs provided.
    - It predicts two outcomes: **"Likely to Default"** or **"Unlikely to Default"**, with a confidence score that reflects the certainty of the prediction.
    - It also suggests potential steps to reduce the risk of default for clients predicted to be at risk.

    ### Why It Matters:
    This tool can help in the assessment of financial stability by businesses, financial institutions, and individuals to take proactive steps in reducing the possibility of defaults, thus improving better financial planning and risk management.
    """)

