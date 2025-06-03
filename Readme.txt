# Loan Approval Prediction App (Rwanda-Specific Prototype)

This interactive app simulates an intelligent loan screening system tailored to the **Rwandan banking sector**. It uses **machine learning** enhanced with **domain-inspired rules** to make explainable, risk-aware lending decisions.

---

## Purpose

This is a **prototype simulation**, demonstrating how credit decisions can be:
- Faster  
- Fairer  
- More transparent  

It is not a production tool yet — but can serve as a **foundation to build robust loan decisioning systems** when co-designed with domain experts in Rwanda’s financial services.

---

## Use Case Fit

Designed for use by:
- Commercial banks
- MFIs and SACCOs
- Digital lending startups
- Credit scoring & analytics teams

### Additional Use:
Bank clients can use this simulation to **pre-evaluate their own eligibility** before applying — reducing in-branch time and unnecessary paperwork.

---

## Lending Logic Based on Local Reality

This app integrates **basic credit rules contextualized for Rwanda**:

- Reject if **monthly income < RWF 120,000**
- Reject if **loan exceeds 40% of annual income**
- Reject if credit grade is **F or G**
- Reject if employment length is **< 1 year**
- Only accept loans with a predicted **repayment probability ? 65%**

---

## Prediction + Explainability

Combines machine learning prediction with **visual explanations** using SHAP:

- Predicts likelihood of full repayment
- Displays a waterfall plot of feature contributions (SHAP)
- Explains *why* an application is accepted or rejected

---

## Tech Stack

| Component       | Technology                     |
|----------------|---------------------------------|
| Frontend        | Streamlit                      |
| Model           | XGBoost / Logistic Regression  |
| Preprocessing   | Scikit-learn Pipelines         |
| Balancing       | SMOTE (imbalanced-learn)       |
| Explainability  | SHAP                           |
| Model Storage   | joblib                         |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-org/loan-approval-rwanda.git
cd loan-approval-rwanda

# Set up environment
python -m venv   

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run loan_approval_app.py
