# SmartLoanAI – AI Powered Loan Approval System

🚀 **Live Demo:** https://smartloanaiml.streamlit.app/

SmartLoanAI is an end-to-end **machine learning–based credit decision system** that predicts whether a loan applicant should be approved or rejected using **Support Vector Machines (SVM)**.

This project is built to simulate how real banks evaluate **credit risk** and make loan decisions.

---

## 🧠 Problem Statement

Banks lose money when loans are approved for risky customers who later default.  
The goal of this project is to use historical applicant data to **predict loan approval** while minimizing financial risk.

This system helps:
- Reduce bad loan approvals  
- Improve credit screening accuracy  
- Automate loan decisions  

---

## 📊 Dataset

The dataset contains historical loan applicant records with features such as:

- Gender, Married, Dependents  
- Education, Self-Employed  
- Applicant Income, Co-Applicant Income  
- Loan Amount, Loan Term  
- Credit History  
- Property Area  

**Target Variable:**  
`Loan_Status` → Approved (1) or Not Approved (0)

Missing values were handled and categorical variables were encoded before training.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- GitHub  
- SVM (Linear, Polynomial, RBF kernels)

---

## ⚙️ Machine Learning Approach

Three SVM models were trained and compared:

| Kernel | Description |
|--------|-------------|
| Linear | Works well for linearly separable data |
| Polynomial | Captures polynomial relationships |
| RBF | Captures complex non-linear patterns |

Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  

---

## 📈 Model Comparison

| Kernel | Accuracy | Risky Customers Approved | Eligible Customers Rejected |
|--------|----------|--------------------------|-----------------------------|
| Linear | 85.3% | 17 | 1 |
| Polynomial | 83.7% | 17 | 3 |
| RBF | 82.9% | **15 (Lowest)** | 6 |

Although Linear had slightly higher accuracy, **RBF was chosen** because it approves **fewer risky customers**, which is critical in financial systems.

---

## 🌐 Web Application

The trained models were deployed using **Streamlit**.  
The app allows users to:

- Enter applicant details  
- Choose SVM kernel (Linear / Polynomial / RBF)  
- Get real-time loan decision  
- View model confidence  
- See risk level visualization  

🔗 **Live App:** https://smartloanaiml.streamlit.app/

---

## 🧠 System Flow

User Input → Preprocessing → SVM Model → Risk Scoring → Decision UI

---

## ▶️ Run Locally

```bash
git clone https://github.com/AshrithaShaganti/SmartLoanAI.git
cd SmartLoanAI
pip install -r requirements.txt
streamlit run app.py
