#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Para iniciar o programa, é necessário importar alguma bibliotecas que são bastante utilizadas quando se trata 
#de machine learning e manipulação dos dados.

import numpy as np #Biblioteca "matemática"
import pandas as pd #Biblioteca para manipulação e análise de dados
import matplotlib.pyplot as plt #Extenção da biblioteca que faz a pltagem de gráficos e pontos
import seaborn as sns
import os #Funcionalidade simplificadas de sistema operacionais


# In[4]:



df=pd.read_csv('measures_v2.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11])


# In[5]:


#Como podemos observar no dataset, existe um grande número de colunas para nossa variável, porém gostariamo de predizer "pm" que é
# a temperatura do rotor. Para isto, colocaremos ela como nosso Target.
df.head()


# In[6]:


#Então pegamos a coluna com os valores de pm e as declaramos como o Target
target = df.pop('pm') #Temperatura do rotor
#Após isto a colocamos é feita a concatenção dela novamente no tabela de dados, porém agora ela se encontra na última coluna de nosso dataframe.
df = pd.concat([df, target], axis=1)
#Aqui é utilizado a função sample para embalhamento dos valores.
df = df.sample(frac=1,random_state=0) #embaralha os dados do dataframe #Ajuda a previnir o overfitting
df.head()


# In[7]:


#Como pode ser visto, o Index de nosso dataframe estava desorganizado. Utilizando o Reset index foi possível traze-lo para a ordem novamente 
#Tendo o index organizado novamente.
df.reset_index(drop=True, inplace=True) #Faz com que o Index volte a ser o que era antes
df.head()


# In[8]:


#Neste momento fazemos a divisão de nosso banco de dados em 75% e 25% a partir do quantidade de linhas de df.
split_index=int(len(df) * 0.75)

#Então aqui fazemos uma jogada bem típica de machine learning onde é separado o conjunto de treino e conjunto de teste.
train_df = df[:split_index] #Primeiros 75%
test_df = df[split_index:] #outros 25% restantes

#É possível observar pela saída dos dados que temos exatamente 25% de linhas para o teste e 
#75% de linhas para o treino de nosso algorítmo.
train_df.info()
test_df.info()


# In[9]:


#Então aqui é colocado os valores de X_train como sendo todas as colunas menos a última, 
#isto é feito para isolar o valor Target que queremos predizer e que havia sido colocado para a última coluna
X_train = train_df.to_numpy()[:, :-1] 
#Aqui é capturada apenas a última coluna para termos o valor o qual o X_train deve alcançar.
y_train = train_df.to_numpy()[:, -1]
#Nesta parte, é feito exatamente a mesma coisa, porém agora sendo utilizado o conjunto de teste.
X_test = test_df.to_numpy()[:, :-1]
y_test = test_df.to_numpy()[:, -1] 


# ### Agora que temos os dados tratados, será instalado o algorítmo TPOT que irá atuar na parte de treinamento do modelo.

# In[10]:


get_ipython().system('pip install tpot')
get_ipython().system('pip install pydataset')


# ### Agora iremos trazer da biblioteca do TPOT a parte do modelo de regressão que ela oferece.
# 
# 

# In[11]:


from tpot import TPOTRegressor


# ### Agora iremos utilizar TPOT Regressor para treinar nosso modelo e encontrar o melhor algorítmo e parâmetros

# In[13]:


#Neste momento estamos colocando os parâmetros ao TPOT
tpot = TPOTRegressor(generations=10,population_size=50,verbosity=2,random_state=5, max_time_mins=180)
#Agora iremos fazer o Fit de nossos dados ao modelo.
tpot.fit(X_train,y_train)
print(tpot.score(X_test,y_test))

