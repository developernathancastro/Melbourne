import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib


pd.set_option('display.max_columns', None)
base = pd.read_csv(r'pages/data/melb_data.csv', sep=',')

#Verificando quantidade de linhas da minha base e cardinalidade entre as feactures
print(f'Quantidade de linhas na base {len(base)}')
for feactures in base:
    print(feactures)
    print(base[feactures].nunique())

#Retirnando feactures
base = base.drop(['Suburb', 'Address', 'SellerG', 'Date', 'BuildingArea', 'CouncilArea' , 'Bedroom2'] , axis=1)

#Verificando quantidade de valores nulos nas feactures e retirando caso necessário
print(base.isna().sum())
base = base.drop(['YearBuilt'], axis=1)

colunas_categorias = ['Type', 'Method', 'Regionname']
base = pd.get_dummies(data=base, columns=colunas_categorias)

print(base.dtypes)
plt.figure(figsize=(15,10))
sns.heatmap(base.corr(), annot=True, cmap='Greens')
plt.show()
#Definição de funções para análise de Outliers

def limites (coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return [q1 - 1.5 * amplitude, q3 + 1.5 *amplitude]
def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_superior = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_superior),:]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas

def diagrama_caixa(coluna):
    fig,(ax1, ax2) = plt.subplots(1, 2)  ##para dois gráficos
    fig.set_size_inches(15, 5)             ##para dois gráficos
    sns.boxplot(x = coluna, ax= ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x = coluna, ax = ax2)
    plt.show()
def histograma(coluna):
    plt.figure(figsize= (15,5)) #-- Para um gráfico
    sns.displot(coluna)
    plt.show()

def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))  # -- Para um gráfico
    ax = sns.barplot(x =coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))
    plt.show()

def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    rsme = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR2:{r2:.2%}\nRSME:{rsme:.2f}'
#Retirando outilers

#Rooms
diagrama_caixa(base['Rooms'])
histograma(base['Rooms'])
base, linhas_removidas = excluir_outliers(base, 'Rooms')
grafico_barra(base['Rooms'])
print(linhas_removidas)

#Distance
diagrama_caixa(base['Distance'])
histograma(base['Distance'])
base, linhas_removidas = excluir_outliers(base, 'Distance')
grafico_barra(base['Distance'])
print(linhas_removidas)

#Postcode
diagrama_caixa(base['Postcode'])
histograma(base['Postcode'])
base, linhas_removidas = excluir_outliers(base, 'Postcode')
grafico_barra(base['Postcode'])
print(linhas_removidas)

#Bathroom
diagrama_caixa(base['Bathroom'])
histograma(base['Bathroom'])
base, linhas_removidas = excluir_outliers(base, 'Bathroom')
grafico_barra(base['Bathroom'])
print(linhas_removidas)

# Car
diagrama_caixa(base['Car'])
histograma(base['Car'])
base, linhas_removidas = excluir_outliers(base, 'Car')
grafico_barra(base['Car'])
print(linhas_removidas)

#Landsize
diagrama_caixa(base['Landsize'])
histograma(base['Landsize'])
base, linhas_removidas = excluir_outliers(base, 'Landsize')
grafico_barra(base['Landsize'])
print(linhas_removidas)

#Propertycount
diagrama_caixa(base['Propertycount'])
histograma(base['Propertycount'])
base, linhas_removidas = excluir_outliers(base, 'Propertycount')
grafico_barra(base['Propertycount'])
print(linhas_removidas)

#Verificando correlação após retirada de outliers
print(base.dtypes)
plt.figure(figsize=(15,10))
sns.heatmap(base.corr(), annot=True, cmap='Greens')
plt.show()

#Vizzualização de mapa das propriedades

amostra = base.sample(n=500)
centro_mapa = {'lat':amostra.Lattitude  .mean(), 'lon':amostra.Longtitude.mean()}
mapa = px.density_mapbox(amostra, lat='Lattitude', lon='Longtitude',z='Price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='stamen-terrain')
mapa.show()

#Fazendo copia de base
base_auxiliar = base.copy()

#separando variais X e Y

y = base['Price']
x = base.drop(['Price'], axis = 1)

#Escolha de modelos a serem testados

modelo_rf= RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_ad = DecisionTreeRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'DecisionTreeRegressor': modelo_ad,
          }

#Separando os dados em treino e teste e treitando modelos

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(x_train,y_train)
    #testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))

#Resultados de modelos treinados e testados

#Modelo RandomForest:
#R2:72.19%
#RSME:310117.98
#Modelo LinearRegression:
#R2:57.29%
#RSME:384287.47
#Modelo DecisionTreeRegressor:
#R2:54.87%
#RSME:395050.66

importancia_features = pd.DataFrame(modelo_rf.feature_importances_, x_train.columns)
importancia_features = importancia_features.sort_values(by= 0 , ascending= False)   ##ordenando coluna
print(importancia_features)

#Retirando features sem muito impacto no modelo e testando novamnete
base_auxiliar = base_auxiliar.drop(['Regionname_Eastern Victoria', 'Regionname_Western Victoria'], axis=1)

#testando modelo sem ela

y = base_auxiliar['Price']
x = base_auxiliar.drop('Price', axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)
print(x_train)

modelo_rf.fit(x_train, y_train)
previsao = modelo_rf.predict(x_test)
print(avaliar_modelo('RandomForest', y_test, previsao))

importancia_features = pd.DataFrame(modelo_rf.feature_importances_, x_train.columns)
importancia_features = importancia_features.sort_values(by= 0 , ascending= False)   ##ordenando coluna
print(importancia_features)

#Deploy do projeto
x['Price'] = y
x.to_csv(r'pages/data/dados.csv')

joblib.dump(modelo_rf, r'pages/model/modelo.joblib')







