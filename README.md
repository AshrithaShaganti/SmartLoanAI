**1. Problem Statement**

Financial institutions face high risk when approving loans for applicants who may default.
The goal of this project is to build a machine learning model that predicts whether a loan applicant is likely to default based on their financial and personal information.

This helps banks:

      Reduce credit risk

      Improve approval accuracy

      Automate decision making

**2. Dataset**

The dataset contains historical loan applicant records with features such as:

      Income

      Loan amount

      Credit history

      Employment status

      Loan term

      Applicant details

Target variable:

      Loan_Status (Approved / Not Approved)

The data was cleaned, missing values were handled, and categorical variables were encoded before modeling.

**3. Tools & Technologies**

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Git & GitHub

**4. Exploratory Data Analysis (EDA)**

EDA was performed to:

    Identify missing values

    Detect outliers

    Analyze feature distributions

    Understand correlations between variables and loan approval

Key insights:

  Credit history and applicant income had strong impact on loan approval

  Applicants with higher income and good credit were more likely to get approved

**5. Feature Engineering**

The following preprocessing steps were applied:

    Missing value imputation

    Label encoding for categorical variables

    Feature scaling where required

These steps improved model stability and performance.

**6. Machine Learning Models Used**

The following models were trained and compared:

    Logistic Regression

    Random Forest Classifier

    Model performance was evaluated using:

    Accuracy

    Precision
  
    Recall

    Confusion Matrix

The best performing model was selected for deployment.

**7. Model Deployment**

The trained model was deployed using Streamlit to create an interactive web application.
Users can input applicant details and get real-time predictions on loan approval.

Live App:
http://smartloanaiml.streamlit.app/

**8. Results**

The final model was able to classify loan approval with good accuracy and reliability, helping simulate a real-world loan screening system.

**9. How to Run Locally**

git clone https://github.com/AshrithaShaganti/<repo-name>
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py

**10. Conclusion**

This project demonstrates an end-to-end machine learning workflow including:

    Data cleaning

    Exploratory data analysis

    Model training and evaluation

Deployment into a real-world web application

It shows how machine learning can be used to solve real financial decision-making problems.
