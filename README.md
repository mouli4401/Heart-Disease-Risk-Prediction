# Heart-Disease-Risk-Prediction

💓 Heart Disease Risk Prediction (10-Year CHD)
📌 Project Overview

This project predicts the 10-year risk of coronary heart disease (CHD) using patient health records.
It is based on the Framingham Heart Study dataset (widely used in medical ML research).
The model takes input features like age, blood pressure, cholesterol, smoking habits, diabetes, BMI, etc., and predicts whether a patient is likely to develop CHD within the next 10 years.

The project includes:

🧠 Machine Learning Model (Logistic Regression with scaling)

🌐 Streamlit Web App for interactive predictions

📊 Patient Summary & Probability Score after prediction

⚙️ Features

Accepts patient details (age, education, sex, smoking, BP, cholesterol, etc.)

Predicts TenYearCHD (0 = Low risk, 1 = High risk)

Displays probability score (e.g., 0.07 = 7% risk)

User-friendly Streamlit interface with a modern layout

Patient summary table to double-check inputs

📊 Dataset

The dataset used is derived from the Framingham Heart Study, containing variables like:

age, education, sex, is_smoking, cigsPerDay

BPMeds, prevalentStroke, prevalentHyp, diabetes

totChol, sysBP, diaBP, BMI, heartRate, glucose

Target: TenYearCHD (1 = disease within 10 years, 0 = no disease)

🚀 Tech Stack

Python (pandas, scikit-learn, pickle)

Streamlit (for interactive web app)

Matplotlib / Seaborn (for evaluation plots)

🖥️ How to Run

Clone this repository:

git clone https://github.com/your-username/heart-disease-risk-predictor.git
cd heart-disease-risk-predictor


Install requirements:

pip install -r requirements.txt


Train the model:

python train_model.py


Run the Streamlit app:

streamlit run app.py

📈 Model Performance

Accuracy: 86%

Precision, Recall, F1, and ROC-AUC also reported (important for imbalanced healthcare datasets).

Example:

Accuracy: 0.86

Precision: 0.80

Recall: 0.78

F1-score: 0.79

🎯 Use Case

This tool can help doctors, researchers, and individuals assess heart disease risk based on simple clinical parameters.
⚠️ Disclaimer: This project is for educational purposes only and should not be used as a replacement for professional medical advice.
