import os
import pandas as pd
import streamlit as st
import time
@st.cache_data
def converte_csv(df):
    return df.to_csv(index=False).encode('utf-8')
def mensagem_sucesso():
        sucesso = st.success('File downloaded successfully!!', icon ="✅")
        time.sleep(5)
        sucesso.empty()

st.title('Database')

dados = pd.read_csv(  r'pages/data/melb_data.csv',sep=',', encoding='utf-8', index_col=0)

st.sidebar.title('Filter')
with st.sidebar.expander('Columns'):
    colunas = st.multiselect('Select the columns', list(dados.columns), list(dados.columns))

dados = dados[colunas]
st.dataframe(dados)

st.markdown(f'The table has :blue[{dados.shape[0]}] rows and :blue[{dados.shape[1]}] columns')

st.markdown('Write a name for the file')

coluna1, coluna2 = st.columns(2)
with coluna1:
    nome_arquivo = st.text_input('', label_visibility='collapsed', value='Data')
    nome_arquivo += '.csv'

with coluna2:
    st.download_button('Download the csv table', data=converte_csv(dados), file_name=nome_arquivo, mime='text/csv', on_click=mensagem_sucesso)



