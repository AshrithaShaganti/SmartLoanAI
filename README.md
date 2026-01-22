# SmartLoanAI – Loan Default & Approval Prediction using Machine Learning

🚀 **Live Demo:** https://smartloanaiml.streamlit.app/

SmartLoanAI is an end-to-end **machine learning–based credit decision system** that predicts whether a loan applicant should be approved or rejected using **Support Vector Machines (SVM)**.

This project simulates how real banks evaluate **credit risk** and make data-driven loan decisions.

---

## 🧠 Problem Statement

Banks face financial loss when high-risk applicants default on loans.  
This project uses historical loan data to predict loan approval and minimize credit risk.

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
- Support Vector Machines (Linear, Polynomial, RBF)  

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to understand feature distributions, detect missing values, and analyze relationships between applicant attributes and loan approval.

**Key insights:**
- Applicants with good credit history were significantly more likely to be approved  
- Higher applicant income correlated with higher approval rates  
- Property area and education also influenced approval probability  

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

This highlights the importance of optimizing for **business risk** rather than raw accuracy.

Although Linear had slightly higher accuracy, **RBF was chosen** because it approves **fewer risky customers**, which is critical in financial systems.

---

## 🌐 Web Application

The trained models were deployed using **Streamlit**.  
The application allows users to:

- Enter applicant details  
- Choose SVM kernel (Linear / Polynomial / RBF)  
- Get real-time loan decision  
- View model confidence  
- See risk level visualization  

🔗 **Live App:** https://smartloanaiml.streamlit.app/

---

## 🧠 System Flow

User Input → Data Preprocessing → SVM Model → Risk Scoring → Decision Interface

---

## ▶️ Run Locally

```bash
git clone https://github.com/AshrithaShaganti/SmartLoanAI.git
cd SmartLoanAI
pip install -r requirements.txt
streamlit run app.py
