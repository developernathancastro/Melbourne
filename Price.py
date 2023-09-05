import streamlit as st
import plotly.express as px
import os
import pandas as pd
from PIL import Image
import joblib

st.set_page_config(layout='wide')
st.title('Project Melbourn')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)
base = pd.read_csv(r'pages/data/dados.csv', sep=',', encoding='utf-8', index_col=0)
print(base)

x_numericos = {'Lattitude': 0, 'Longtitude': 0, 'Landsize': 0, 'Propertycount': 0, 'Rooms': 0,
               'Distance': 0, 'Postcode': 0, 'Bathroom': 0, 'Car': 0}

x_listas = {'Regionname': ['Eastern Metropolitan', 'Northern Metropolitan', 'Northern Victoria', 'South-Eastern Metropolitan',
                           'Southern Metropolitan', 'Western Metropolitan'],
            'Method': ['PI', 'S', 'SA', 'SP', 'VB'],
            'Type': ['h', 't', 'u']}

dicionario = {}

for item in x_listas:
    for valor in x_listas[item]:
        dicionario[f'{item}_{valor}'] = 0

for item in x_numericos:
    if item == 'Lattitude' or item == 'Longtitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format='%5f')

    elif item == 'Distance':
        valor = st.number_input(f'{item}', step=0.5, value=1.0)
        x_numericos[item] = valor

    else:
        valor = st.number_input(f'{item}', step=1, value=0)
        x_numericos[item] = valor

for item in x_listas:
    valor = st.selectbox(f'{item}', x_listas[item])
    dicionario[f'{item}_{valor}'] = 1

botao = st.button('Predict Property Value')

if botao:
    dicionario.update(x_numericos)
    valores_x = pd.DataFrame(dicionario, index=[0])
    valores_x = valores_x[['Rooms', 'Distance', 'Postcode', 'Bathroom', 'Car', 'Landsize',
                          'Lattitude', 'Longtitude', 'Propertycount', 'Type_h', 'Type_t',
                          'Type_u', 'Method_PI', 'Method_S', 'Method_SA', 'Method_SP',
                          'Method_VB', 'Regionname_Eastern Metropolitan',
                          'Regionname_Northern Metropolitan', 'Regionname_Northern Victoria',
                          'Regionname_South-Eastern Metropolitan',
                          'Regionname_Southern Metropolitan', 'Regionname_Western Metropolitan']]
    modelo = joblib.load(r'pages/model/modelo.joblib')
    preco = modelo.predict(valores_x)
    st.write(f'Expected value of the property: {preco[0]}')


































