# ❤️ Heart Disease Risk Prediction (End-to-End ML Project)

An end-to-end Machine Learning project that predicts the risk of heart disease based on personal health metrics and lifestyle habits, deployed as an interactive web application using **Streamlit**.

## 📊 Dataset Overview
* **Records:** ~300,000+ entries with 18 features (BMI, Smoking, Stroke, PhysicalHealth, AgeCategory, GenHealth, etc.).
* **Challenge:** Highly imbalanced dataset (~9% positive cases).

## 🛠️ Methodology & Key Decisions
1. **Exploratory Data Analysis (EDA):** Identified class imbalance, strong correlation between chronic diseases/age and heart disease, and significant outliers in numerical features (`BMI`, `SleepTime`, `PhysicalHealth`).
2. **Preprocessing:**
   * **Encoding:** Binary mapping (`Yes/No`), Ordinal encoding (`AgeCategory`, `GenHealth`), and One-Hot Encoding (`Race`, `Diabetic`).
   * **Splitting:** Stratified 80/20 train-test split to preserve the minority class ratio.
   * **Scaling:** Used `RobustScaler` to handle outliers effectively.
3. **Modeling:**
   * Compared **Logistic Regression** and **Random Forest** using `class_weight='balanced'`.
   * Selected **Logistic Regression** as the final model because it achieved a high **Recall (0.77)** and **ROC-AUC (0.8324)**, minimizing missed positive cases—which is critical in medical diagnosis.

## 📁 Project Structure
* `data/heart_2020_cleaned.csv`: Dataset file.
* `models/heart_disease_model.pkl`: Saved Logistic Regression model and RobustScaler.
* `notebooks/heart_disease_ml.ipynb`: EDA, preprocessing, and model training notebook.
* `app.py`: Streamlit web application.
* `requirements.txt`: Required Python libraries.

## 🚀 How to Run Locally
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the Streamlit app: `python -m streamlit run app.py`

---
**Prepared by:** Mohamed Mahmoud Hemdan