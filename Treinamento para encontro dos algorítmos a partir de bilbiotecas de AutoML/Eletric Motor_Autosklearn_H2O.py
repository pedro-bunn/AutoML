#!/usr/bin/env python
# coding: utf-8

# # # EDA And Baseline on Motor Temperature Estimation With H2O
# 

# # Dependências básicas

# In[1]:


get_ipython().run_line_magic('pip', 'install seaborn')
get_ipython().run_line_magic('pip', 'install requests')
get_ipython().run_line_magic('pip', 'install tabulate')
get_ipython().run_line_magic('pip', 'install future')


# In[2]:


get_ipython().run_line_magic('pip', 'install h2o')


# ### Importando Bibliotecas

# In[3]:


import numpy as np #Biblioteca "matemática"
import pandas as pd #Biblioteca para manipulação e análise de dados
import matplotlib.pyplot as plt #Extenção da biblioteca que faz a pltagem de gráficos e pontos
import seaborn as sns
import os #Funcionalidade simplificadas de sistema operacionais
print(os.listdir())


# ### Leitura e tratamento do dataset

# In[4]:


df=pd.read_csv('measures_v2.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
target = df.pop('pm') #Temperatura do rotor
df = pd.concat([df, target], axis=1)
df = df.sample(frac=1,random_state=0) #embaralha os dados do dataframe #Ajuda a previnir o overfitting
df.reset_index(drop=True, inplace=True) #Faz com que o Index volte a ser o que era antes
df


# ### Separação dos dados de treino e dados de teste

# In[5]:


split_index=int(len(df) * 0.75)

train_df = df[:split_index] #Primeiros 75%
test_df = df[split_index:] #outros 25% restantes

train_df.info()
test_df.info()


# #### Separação da última linha em dados de treino e dados de teste

# In[6]:


X_train = train_df.to_numpy()[:, :-1] 
y_train = train_df.to_numpy()[:, -1]

X_test = test_df.to_numpy()[:, :-1]
y_test = test_df.to_numpy()[:, -1] 


# ## Seguindo instalação segundo diretrizes do H2O

# ### Teste do biblioteca

# In[7]:


import h2o
from h2o.automl import H2OAutoML
h2o.init()


# ### Transformando o Frame para o formato da biblioteca H2O
# 

# In[8]:


h2o_frame=h2o.H2OFrame(train_df)


# ### Identificando os valores das características e o que é necessário predizer

# In[9]:


X=h2o_frame.columns
y='pm'
X.remove(y)


# ### Encontrando os melhores algorítmos a partir da biblioteca H2O

# In[ ]:



aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=X, y=y, training_frame=h2o_frame)


# ### Fazendo as predições de treino e teste com o algorítmos encontrado

# In[ ]:


pred_train=aml.predict(h2o_frame)


# In[ ]:


h2o_frame_test=h2o.H2OFrame(test_df)


# In[ ]:


pred_train=aml.predict(h2o_frame_test)

