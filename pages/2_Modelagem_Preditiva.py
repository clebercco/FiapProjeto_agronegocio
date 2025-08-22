# 2_Modelagem_Preditiva.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_generation import generate_data  # Carrega o dataset
# Configurações gerais
st.set_page_config(page_title='Modelagem Preditiva', layout='wide')
# Título do aplicativo
st.title('Modelagem Preditiva')
# ================================================
# 1. Carregar e Filtrar os Dados
# ================================================
st.header('1. Carregar e Filtrar os Dados')
# Carregar os dados gerados
df = generate_data()
# ----------------------------------
# 1.1. Filtros na Barra Lateral
# ----------------------------------
st.sidebar.title('Filtros de Dados')
# Filtro para Fertilizante
selected_fertilizer = st.sidebar.multiselect(
    'Selecione o Fertilizante:',
    options=df['Fertilizante'].unique(),
    default=df['Fertilizante'].unique()
)
# Filtro para Tipo de Solo
selected_soil = st.sidebar.multiselect(
    'Selecione o Tipo de Solo:',
    options=df['Tipo de Solo'].unique(),
    default=df['Tipo de Solo'].unique()
)
# Filtros para variáveis numéricas
st.sidebar.subheader('Intervalos das Variáveis Numéricas')
# Temperatura
temp_min, temp_max = st.sidebar.slider(
    'Temperatura (°C):',
    min_value=float(df['Temperatura'].min()),
    max_value=float(df['Temperatura'].max()),
    value=(float(df['Temperatura'].min()), float(df['Temperatura'].max()))
)
# Precipitação
precip_min, precip_max = st.sidebar.slider(
    'Precipitação (mm):',
    min_value=float(df['Precipitação'].min()),
    max_value=float(df['Precipitação'].max()),
    value=(float(df['Precipitação'].min()), float(df['Precipitação'].max()))
)
# Umidade
umid_min, umid_max = st.sidebar.slider(
    'Umidade (%):',
    min_value=float(df['Umidade'].min()),
    max_value=float(df['Umidade'].max()),
    value=(float(df['Umidade'].min()), float(df['Umidade'].max()))
)
# ----------------------------------
# 1.2. Aplicar Filtros aos Dados
# ----------------------------------
# Aplicar os filtros selecionados ao DataFrame
df_filtered = df[
    (df['Fertilizante'].isin(selected_fertilizer)) &amp;
    (df['Tipo de Solo'].isin(selected_soil)) &amp;
    (df['Temperatura'] >= temp_min) &amp; (df['Temperatura'] <= temp_max) &amp;
    (df['Precipitação'] >= precip_min) &amp; (df['Precipitação'] <= precip_max) &amp;
    (df['Umidade'] >= umid_min) &amp; (df['Umidade'] <= umid_max)
]
# Verificar se o DataFrame filtrado não está vazio
if df_filtered.empty:
    st.warning('Nenhum dado corresponde aos filtros selecionados. Por favor, ajuste os filtros.')
    st.stop()
else:
    st.subheader('Dados Filtrados')
    st.dataframe(df_filtered.head())
# ================================================
# 2. Preparar os Dados para Modelagem
# ================================================
st.header('2. Preparar os Dados para Modelagem')
# Transformar variáveis categóricas em variáveis dummies
df_ml = pd.get_dummies(df_filtered, columns=['Fertilizante', 'Tipo de Solo'])
# Separar as variáveis independentes (X) e a variável dependente (y)
X = df_ml.drop('Produção', axis=1)
y = df_ml['Produção']
# Verificar se há dados suficientes para treinar o modelo
if len(X) < 2:
    st.warning('Dados insuficientes para treinar o modelo. Por favor, ajuste os filtros para incluir mais dados.')
    st.stop()
else:
    st.write(f'**Total de registros após filtragem:** {len(X)}')
# ================================================
# 3. Treinar o Modelo de Machine Learning
# ================================================
st.header('3. Treinar o Modelo de Machine Learning')
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Instanciar o modelo de Random Forest Regressor
model = RandomForestRegressor()
# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)
# Avaliar o modelo com os dados de teste
score = model.score(X_test, y_test)
st.write(f'**Acurácia do modelo (R² no conjunto de teste):** {score:.2f}')