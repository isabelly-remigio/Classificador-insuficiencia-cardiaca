import streamlit as st
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

modelo = joblib.load('modelo_random_forest.pkl')
df = pd.read_csv('heart.csv')


def encode_inputs(sex, chest_pain_type, resting_ecg, exercise_angina, st_slope):
    sex_dict = {'M': 1, 'F': 0}
    chest_pain_dict = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3} 
    resting_ecg_dict = {'LVH': 0, 'Normal': 1, 'ST': 2}
    exercise_angina_dict = {'N': 0, 'Y': 1}
    st_slope_dict = {'Down': 0, 'Flat': 1, 'Up': 2}

    return [
        sex_dict.get(sex, 0),
        chest_pain_dict.get(chest_pain_type, 0),
        resting_ecg_dict.get(resting_ecg, 0),
        exercise_angina_dict.get(exercise_angina, 0),
        st_slope_dict.get(st_slope, 0)
    ]


modelo = joblib.load('modelo_random_forest.pkl')


st.sidebar.header("Insira os valores abaixo:")

age = st.sidebar.number_input('Idade (anos)', min_value=0, max_value=120, value=40)
sex = st.sidebar.selectbox('Sexo', options=['M', 'F'])
chest_pain_type = st.sidebar.selectbox('Tipo de Dor no Peito', options=['ASY', 'ATA', 'NAP', 'TA'])
resting_bp = st.sidebar.number_input('Pressão Arterial em Repouso (mmHg)', min_value=0, max_value=200, value=140)
cholesterol = st.sidebar.number_input('Colesterol (mg/dL)', min_value=0, max_value=600, value=289)
fasting_bs = st.sidebar.selectbox('Glicose em Jejum', options=[0, 1], format_func=lambda x: 'Normal' if x == 0 else 'Alterada')
resting_ecg = st.sidebar.selectbox('ECG em Repouso', options=['Normal', 'ST', 'LVH'])
max_hr = st.sidebar.number_input('Frequência Cardíaca Máxima (bpm)', min_value=0, max_value=220, value=172)
exercise_angina = st.sidebar.selectbox('Angina de Exercício', options=['N', 'Y'])
oldpeak = st.sidebar.number_input('Oldpeak (Depressão ST)', min_value=0.0, max_value=10.0, value=0.0)
st_slope = st.sidebar.selectbox('Inclinação do Segmento ST', options=['Up', 'Flat', 'Down'])


encoded_inputs = encode_inputs(sex, chest_pain_type, resting_ecg, exercise_angina, st_slope)


input_data = np.array([
    age,
    encoded_inputs[0], 
    encoded_inputs[1],  
    resting_bp,
    cholesterol,
    fasting_bs,
    encoded_inputs[2], 
    max_hr,
    encoded_inputs[3],  
    oldpeak,
    encoded_inputs[4]   
])


if st.sidebar.button("Prever"):
    prediction = modelo.predict([input_data])[0]
    
    if prediction == 1:
        st.success("**Resultado: A pessoa é hipertensa.**")
    else:
        st.info("**Resultado: A pessoa não é hipertensa.**")




if st.checkbox("Mostrar gráfico de importância das variáveis"):
    st.subheader("Importância das Variáveis no Modelo")

    
    PREDITORAS = df.drop(columns=['HeartDisease'])  
    feature_names = PREDITORAS.columns
    importances = modelo.feature_importances_

    df_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_importances, palette="viridis", ax=ax)
    ax.set_title("Importância das Features - Random Forest")
    ax.set_xlabel("Importância")
    ax.set_ylabel("Variável")
    st.pyplot(fig)




if st.checkbox("Matriz de Correlação entre Variáveis"):
    st.subheader("Matriz de Correlação entre Variáveis")


    correlation_matrix = df.corr(numeric_only=True)


    fig, ax = plt.subplots(figsize=(12, 8))


    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Correlação entre Variáveis")


    st.pyplot(fig)