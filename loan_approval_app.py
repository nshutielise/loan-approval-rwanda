import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ---------------------
# 🎯 Page Setup
# ---------------------
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.markdown("Provide loan applicant details to predict approval likelihood using a trained ML model with SMOTE balancing.")

# ---------------------
# 📅 Input Fields
# ---------------------
loan_amount = st.number_input("Loan Amount (RWF)", min_value=0, value=8000, step=500)
annual_income = st.number_input("Annual Income (RWF)", min_value=0, value=80000, step=1000)
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=11.5, step=0.1)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0, step=0.5)

emp_length = st.selectbox("Employment Length", [
    '< 1 year', '1 year', '2 years', '3 years', '4 years',
    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
])
purpose = st.selectbox("Loan Purpose", [
    'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'
])
grade = st.selectbox("Credit Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN'])

# ---------------------
# 🗒️ Prepare Input DataFrame
# ---------------------
input_df = pd.DataFrame([{
    'loan_amount': loan_amount,
    'annual_income': annual_income,
    'int_rate': int_rate,
    'dti': dti,
    'emp_length': emp_length,
    'purpose': purpose,
    'grade': grade,
    'home_ownership': home_ownership
}])

st.markdown("----")

# ---------------------
# 🧠 Load the Trained Model
# ---------------------
model = None
model_path = "logreg_model_compatible.pkl"

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.error("❌ Model file not found. Ensure 'logreg_model_compatible.pkl' is in the current directory.")

# ---------------------
# 📈 Make Prediction
# ---------------------
if st.button("🔍 Predict Loan Approval"):
    if model is not None and hasattr(model, "predict"):
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            # ---------------------
            # ⚖️ Updated Rwandan Lending Rules
            # ---------------------
            monthly_income = annual_income / 12
            annual_income_estimated = monthly_income * 12
            max_loan_allowed = 0.4 * annual_income_estimated

            risk_flag = False
            reasons = []

            if monthly_income < 120000:
                risk_flag = True
                reasons.append("Monthly income is below RWF 120,000")

            if loan_amount > max_loan_allowed:
                risk_flag = True
                reasons.append("Loan exceeds 40% of annual income")

            if grade in ['F', 'G']:
                risk_flag = True
                reasons.append("Very low credit grade")

            if emp_length == "< 1 year":
                risk_flag = True
                reasons.append("Insufficient employment history")

            # ---------------------
            # ✅ Final Decision
            # ---------------------
            if risk_flag:
                st.error(f"❌ Loan Rejected based on policy: {', '.join(reasons)}")
            elif probability >= 0.65:
                st.success(f"✅ Loan Approved with {probability:.2%} probability.")
            else:
                st.warning(f"⚠️ Loan Not Approved: Probability too low ({probability:.2%})")

            st.markdown("#### 🔍 Prediction Details")
            st.json(input_df.to_dict(orient="records")[0])

            # ---------------------
            # 🧠 SHAP Explanation
            # ---------------------
            try:
                with st.spinner("Generating explanation..."):
                    preprocessed = model.named_steps['preprocessor'].transform(input_df)
                    explainer = shap.Explainer(model.named_steps['classifier'], feature_names=model.named_steps['preprocessor'].get_feature_names_out())
                    shap_values = explainer(preprocessed)

                    st.markdown("#### 🔎 Feature Contribution (SHAP)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Could not generate SHAP plot: {e}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.error("⚠️ Model not loaded or invalid.")
