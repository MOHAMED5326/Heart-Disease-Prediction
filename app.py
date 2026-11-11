import streamlit as st
import pandas as pd
import joblib

# 1. Load Model, Scaler, and Columns
model_data = joblib.load('models/heart_disease_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
model_columns = model_data['columns']

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Risk Prediction")
st.write("أدخل البيانات الصحية للمريض للتنبؤ باحتمالية الإصابة بمرض القلب.")

# 2. User Inputs
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex (الجنس)", ["Male", "Female"])
    age_cat = st.selectbox("Age Category (الفئة العمرية)", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])
    bmi = st.number_input("BMI (مؤشر كتلة الجسم)", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    sleep_time = st.number_input("Sleep Time (ساعات النوم)", min_value=1.0, max_value=24.0, value=7.0, step=1.0)
    physical_health = st.slider("Physical Health (أيام التعب الجسدي في آخر 30 يوم)", 0, 30, 0)
    mental_health = st.slider("Mental Health (أيام التعب النفسي في آخر 30 يوم)", 0, 30, 0)
    gen_health = st.selectbox("General Health (الصحة العامة)", ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'], index=2)
    race = st.selectbox("Race (العرق)", ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other'])

with col2:
    smoking = st.selectbox("Smoking (مدخن؟)", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Drinking (شرب الكحول؟)", ["No", "Yes"])
    stroke = st.selectbox("Stroke (جلطة سابقة؟)", ["No", "Yes"])
    diff_walking = st.selectbox("Difficulty Walking (صعوبة في المشي؟)", ["No", "Yes"])
    diabetic = st.selectbox("Diabetic (مرض السكر؟)", ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"])
    physical_activity = st.selectbox("Physical Activity (نشاط بدني؟)", ["Yes", "No"])
    asthma = st.selectbox("Asthma (ربو؟)", ["No", "Yes"])
    kidney_disease = st.selectbox("Kidney Disease (أمراض كلى؟)", ["No", "Yes"])
    skin_cancer = st.selectbox("Skin Cancer (سرطان جلد؟)", ["No", "Yes"])

# 3. Preprocessing & Prediction
if st.button("Predict (توقع النتيجة)", type="primary"):
    input_dict = {
        'BMI': bmi,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'AlcoholDrinking': 1 if alcohol == 'Yes' else 0,
        'Stroke': 1 if stroke == 'Yes' else 0,
        'PhysicalHealth': float(physical_health),
        'MentalHealth': float(mental_health),
        'DiffWalking': 1 if diff_walking == 'Yes' else 0,
        'Sex': 1 if sex == 'Male' else 0,
        'AgeCategory': {
            '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
            '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9,
            '70-74': 10, '75-79': 11, '80 or older': 12
        }[age_cat],
        'PhysicalActivity': 1 if physical_activity == 'Yes' else 0,
        'GenHealth': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}[gen_health],
        'SleepTime': sleep_time,
        'Asthma': 1 if asthma == 'Yes' else 0,
        'KidneyDisease': 1 if kidney_disease == 'Yes' else 0,
        'SkinCancer': 1 if skin_cancer == 'Yes' else 0,
        f'Race_{race}': 1,
        f'Diabetic_{diabetic}': 1
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk:** احتمالية وجود مرض قلب (نسبة الخطورة: {probability:.1%})")
    else:
        st.success(f"✅ **Low Risk:** احتمالية منخفضة للإصابة بمرض القلب (نسبة الخطورة: {probability:.1%})")