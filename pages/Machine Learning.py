import joblib
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
def formatar_valores(valor):
    if valor >= 1e12:
        return f'${valor / 1e12:.2f} Q'
    elif valor >= 1e9:
        return f'${valor / 1e9:.2f} T'
    elif valor >= 1e6:
        return f'${valor / 1e6:.2f} B'
    else:
        return f'${valor:.2f}'
st.set_page_config(layout='wide' )

#Importando base e retirando price
pd.set_option('display.max_columns', None)
dados_previsao = pd.read_csv(r'pages/data/dados.csv',  sep=',', encoding='utf-8', index_col=0)
melb_data = pd.read_csv(r'pages/data/melb_data.csv',  sep=',', encoding='utf-8', index_col=0)

dados_previsao =dados_previsao.drop(['Price'], axis=1)

##importando modelo
modelo = joblib.load(r'pages/model/modelo.joblib')

#fazendo previsões
previsoes = modelo.predict(dados_previsao)
dados_previsao['Forecasts'] = previsoes

st.title('Real Estate Value of the City :shopping_trolley:')

# Transformar colunas de regiões em uma única coluna
regioes = [
    'Regionname_Eastern Metropolitan',
    'Regionname_Northern Metropolitan',
    'Regionname_Northern Victoria',
    'Regionname_South-Eastern Metropolitan',
    'Regionname_Southern Metropolitan',
    'Regionname_Western Metropolitan'
]

dados_previsao['Regioes'] = dados_previsao[regioes].apply(
    lambda row: ', '.join([col.replace('Regionname_', '') for col in regioes if row[col]]),
    axis=1
)

dados_previsao['Regioes'] = dados_previsao[regioes].apply(
    lambda row: ', '.join([col.replace('Regionname_', '') for col in regioes if row[col]]),
    axis=1
)

tipos_de_propriedade = ['Type_h', 'Type_t', 'Type_u']

dados_previsao['Tipo de Propriedade'] = dados_previsao[tipos_de_propriedade].apply(
    lambda row: ', '.join([col.replace('Type_', '') for col in tipos_de_propriedade if row[col]]),
    axis=1
)

metodos_de_venda = ['Method_SA', 'Method_SP', 'Method_VB', 'Method_PI', 'Method_S']

dados_previsao['Método de Venda'] = dados_previsao[metodos_de_venda].apply(
    lambda row: ', '.join([col.replace('Method_', '') for col in metodos_de_venda if row[col]]),
    axis=1
)
#Tabelas
mercado = dados_previsao.groupby('Regioes')[['Forecasts']].sum().sort_values(by='Forecasts', ascending=True).reset_index()

tipo_imovel = dados_previsao.groupby('Tipo de Propriedade')[['Forecasts']].sum().sort_values(by='Forecasts', ascending=True).reset_index()

distancia_cm = dados_previsao[['Distance', 'Forecasts', 'Regioes']].sort_values(by='Forecasts', ascending=True)

metodo_venda = dados_previsao.groupby('Método de Venda')[['Forecasts']].sum().sort_values(by='Forecasts', ascending=True).reset_index()

valor_vendasregiao = melb_data.groupby('Regionname')[['Price']].sum().sort_values(by='Price', ascending=False).reset_index()
vendas_tipoimovel =  melb_data.groupby('Type')[['Price']].sum().sort_values(by='Price', ascending=False).reset_index()

melb_data['Date'] = pd.to_datetime(melb_data['Date'], format = '%d/%m/%Y')
melb_data['Ano'] = melb_data['Date'].dt.year
vendas_ano = melb_data.groupby('Ano')[['Price']].sum().sort_values(by='Price', ascending=True).reset_index()
vendas_metodo = melb_data.groupby('Method')[['Price']].sum().sort_values(by='Price', ascending=True).reset_index()

#Criando gráficos

fig_valormercado = px.bar(mercado,
                          x = 'Regioes',
                          y = 'Forecasts',
                          text_auto=True,
                          title='Market value by region')
fig_valormercado.update_layout(yaxis_title = 'Market Value')
fig_valormercado.update_layout(xaxis_title = 'Region')

fig_tipo_imovel = px.bar(tipo_imovel,
                      x='Tipo de Propriedade',
                      y='Forecasts',
                      text_auto=True,
                    title='Market value by type of property')

fig_tipo_imovel.update_layout(xaxis_title = 'Type of property')

fig_distanciacm = px.scatter(distancia_cm,
                         x='Distance',
                         y='Forecasts',
                         color='Regioes',
                         title='Relationship between distance from CM and market value',
                         labels={'Distance': 'Distance', 'Forecasts': 'Market Valueo'},
                         marginal_x='histogram',
                         marginal_y='histogram',
                         trendline='ols'
                         )

fig_metodo_venda = px.pie(metodo_venda,
                                values='Forecasts',
                                names='Método de Venda',
                                title='Distribution of market value by method of sale')

fig_valorvendasregiao = go.Figure(go.Funnel(
    y = valor_vendasregiao['Regionname'],
    x = valor_vendasregiao['Price']))

fig_valorvendasregiao.update_layout(title= 'Sales by region')

fig_valorvendas_tipoimovel =   px.bar(vendas_tipoimovel,
                          x = 'Type',
                          y = 'Price',
                          text_auto=True,
                          title='Sales by type of property')
fig_valorvendas_tipoimovel.update_layout(yaxis_title = 'Price')

fig_vendasano = px.pie(vendas_ano,
                                values='Price',
                                names='Ano',
                                title='Sales year')

fig_vendasmetodo = px.bar(vendas_metodo,
                          x = 'Price',
                          y = 'Method',
                          text_auto=True,
                          title='Sales by method')

#Visualização no streamlit
aba1, aba2 = st.tabs(['Machine Learning', 'DATA ANALYSIS'])

with aba1:
    coluna1, coluna2 = st.columns(2)
    with coluna1:
        st.metric('Total Market Value',formatar_valores(dados_previsao['Forecasts'].sum()))
        st.plotly_chart(fig_valormercado, use_container_width=True)
        st.plotly_chart(fig_tipo_imovel, use_container_width=True)

    with coluna2:
        st.metric('Total Properties', dados_previsao['Forecasts'].count())
        st.plotly_chart(fig_distanciacm, use_container_width=True)
        st.plotly_chart(fig_metodo_venda, use_container_width=True)

with aba2:
    coluna1, coluna2 = st.columns(2)
    with coluna1:
        st.metric('Total Sales Value', formatar_valores(melb_data['Price'].sum()))
        st.plotly_chart(fig_valorvendasregiao, use_container_width=True)
        st.plotly_chart(fig_vendasano, use_container_width=True)

    with coluna2:
        st.metric('Number of sales in the city', melb_data['Address'].count())
        st.plotly_chart(fig_valorvendas_tipoimovel, use_container_width=True)
        st.plotly_chart(fig_vendasmetodo, use_container_width=True)









































