#!/usr/bin/env python
# coding: utf-8

# ## EDA And Baseline on Motor Temperature Estimation

# In[1]:


get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn==0.23.2 ')


# Importando Bibliotecas

# In[2]:


import numpy as np #Biblioteca "matemática"
import pandas as pd #Biblioteca para manipulação e análise de dados
import matplotlib.pyplot as plt #Extenção da biblioteca que faz a pltagem de gráficos e pontos
from matplotlib.colors import rgb2hex
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import os #Funcionalidade simplificadas de sistema operacionais
from sklearn.model_selection import cross_val_score # Cross Validation Function.
from sklearn.model_selection import KFold # KFold Class.
from sklearn.linear_model import LinearRegression # Linear Regression class.
from sklearn.metrics import mean_squared_error
import sklearn.metrics

print(os.listdir())
plt.style.use('bmh')


# Lendo o arquivo

# In[3]:


df=pd.read_csv('measures_v2.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
target = df.pop('pm') #Temperatura do rotor
df = pd.concat([df, target], axis=1)
df = df.sample(frac=1,random_state=0) #embaralha os dados do dataframe #Ajuda a previnir o overfitting
df.reset_index(drop=True, inplace=True) #Faz com que o Index volte a ser o que era antes


# In[4]:


df.head()


# In[5]:


split_index=int(len(df) * 0.75)

train_df = df[:split_index] #Primeiros 75%
test_df = df[split_index:] #outros 25% restantes

train_df.info()
test_df.info()



# Retira a última coluna que é no target do modelo de treinamento e modelos de teste

# In[6]:


X_train = train_df.to_numpy()[:, :-1] 
y_train = train_df.to_numpy()[:, -1]

X_test = test_df.to_numpy()[:, :-1]
y_test = test_df.to_numpy()[:, -1] 


# Criando o modelo para o treinamento do algorítmo

# In[7]:


knn_model= neighbors.KNeighborsRegressor(n_neighbors=71, p=1, weights='distance')

knn_treino=knn_model.fit(X_train,y_train)
knn_teste=knn_model.fit(X_train,y_train)


# # Fazendo as predições dos valores

# In[8]:


Pred_train_y =knn_model.predict(X_train)


# In[9]:


Pred_test_y =knn_model.predict(X_test)


# Observando o resultado das predições a partir do R^2 e o Mean Squared Error

# In[10]:



print("Scores R2 de treino", sklearn.metrics.r2_score(y_train,Pred_train_y))
print("Scores R2 de teste", sklearn.metrics.r2_score(y_test,Pred_test_y))


# In[11]:


MSE_treino=sklearn.metrics.mean_squared_error(y_train, Pred_train_y)

MSE_teste=sklearn.metrics.mean_squared_error(y_test, Pred_test_y)

print("Erro quadrático Médio Treino", MSE_treino)

print("Erro quadrático Médio Teste", MSE_teste)


# In[12]:


#Dados de treino
ax1 = sns.distplot(y_train, hist=False, color="r", label="Valor real")
sns.distplot(Pred_train_y, hist=False, color="b", label="Valor do treino" , ax=ax1);


# Dados de Teste

# In[13]:


ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(Pred_test_y, hist=False, color="b", label="Fitted Values" , ax=ax1)


# # Dados do treino

# In[14]:


plt.scatter(Pred_train_y,Pred_train_y-  y_train,c='blue', marker='o', label='Training data')
plt.xlabel('Valores preditos')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=110, lw=2, color='red')
plt.xlim([0, 110])
plt.show()


# Dados do treino e teste

# In[15]:


plt.scatter(Pred_train_y,Pred_train_y-  y_train,c='blue', marker='o', label='Training data')
plt.scatter(Pred_test_y, Pred_test_y - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Valores preditos')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=110, lw=2, color='red')
plt.xlim([0, 110])
plt.show()


# # 1. Validação Cruzada

# In[16]:


df=pd.read_csv('measures_v2.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
target = df.pop('pm') #Temperatura do rotor
df = pd.concat([df, target], axis=1)
X_train = train_df.to_numpy()[:, :-1] 
y_train = train_df.to_numpy()[:, -1]
X = test_df.to_numpy()[:, :-1]
y = test_df.to_numpy()[:, -1] 



# In[17]:


knn_model= neighbors.KNeighborsRegressor(n_neighbors=73, p=1, weights='distance')
kfold  = KFold(n_splits=10, shuffle=True) # shuffle=True, Shuffle (embaralhar) the data.
result = cross_val_score(knn_model, X, y, cv = kfold)
rmse_cv=np.sqrt(-1*cross_val_score(knn_model,X,y ,cv=kfold,scoring='neg_mean_squared_error').mean())


print("K-Fold (R^2) Scores: {0}".format(result))
print("Média do R^2 para a validação cruzada K-Fold: {0}".format(result.mean()))
print("Erro quadrático médio", rmse_cv)


