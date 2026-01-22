import streamlit as st
import numpy as np
import joblib

# ---------------- Page config ----------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

# ---------------- Load external CSS ----------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- Hero Header ----------------
st.markdown("""
<div class="hero">
    <h1>🏦Smart Loan Approval System</h1>
    <p>AI-powered credit decisioning using Support Vector Machines</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load models ----------------
svm_linear = joblib.load("svm_linear.pkl")
svm_poly   = joblib.load("svm_poly.pkl")
svm_rbf    = joblib.load("svm_rbf.pkl")
scaler     = joblib.load("scaler.pkl")

# ---------------- Input Section ----------------
st.markdown("<div class='section-title' style='font-size:26px; font-weight:800;'>Enter Applicant Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Identity")

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    married = st.selectbox("Married", ["Yes", "No"])
    married = 1 if married == "Yes" else 0

    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    dependents = {"0":0, "1":1, "2":2, "3+":3}[dependents]

    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    education = 1 if education == "Graduate" else 0

    self_emp = st.selectbox("Self Employed", ["Yes", "No"])
    self_emp = 1 if self_emp == "Yes" else 0

with col2:
    st.markdown("### 💰 Financial")

    income = st.number_input("Applicant Income", min_value=0)
    co_income = st.number_input("Co-Applicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (months)", value=360)

    st.markdown("### 🧾 Credit")

    credit = st.selectbox("Credit History", ["Yes", "No"])
    credit = 1 if credit == "Yes" else 0

    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    property_area = {"Urban":2, "Semiurban":1, "Rural":0}[property_area]

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Kernel Selection ----------------
st.markdown("<div class='section-title'>Select SVM Kernel</div>", unsafe_allow_html=True)

kernel = st.radio("", ["Linear", "Polynomial", "RBF"], horizontal=True)


model = svm_linear if kernel=="Linear" else svm_poly if kernel=="Polynomial" else svm_rbf

# ---------------- Prediction ----------------
if st.button("Check Loan Eligibility"):

    X = np.array([[gender, married, dependents, education, self_emp,
                   income, co_income, loan_amount,
                   loan_term, credit, property_area]])

    X = scaler.transform(X)

    pred = model.predict(X)[0]
    confidence = abs(model.decision_function(X)[0])

    st.markdown("<div class='section-title'>Loan Decision</div>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown("<div class='approved'>✅ Loan Approved</div>", unsafe_allow_html=True)
        st.write("The applicant has a strong credit profile and income pattern, making repayment likely.")
    else:
        st.markdown("<div class='rejected'>❌ Loan Rejected</div>", unsafe_allow_html=True)
        st.write("The applicant shows financial risk indicators, increasing the probability of default.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"<div class='metric'><b>Kernel</b><br>{kernel}</div>", unsafe_allow_html=True)
    with colB:
        st.markdown(f"<div class='metric'><b>Confidence</b><br>{round(confidence,2)}</div>", unsafe_allow_html=True)

    # ---------------- Risk Meter ----------------
    risk_score = min(max(confidence * 50, 0), 100)

    if pred == 1:
        risk_color = "#22c55e"
        risk_text = "Low Risk"
    else:
        risk_color = "#ef4444"
        risk_text = "High Risk"

    st.markdown(f"""
    <div class="risk-container">
        <div class="risk-label">Risk Level: {risk_text}</div>
        <div class="risk-bar">
            <div class="risk-fill" style="width:{risk_score}%; background:{risk_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
