#!/usr/bin/env python
# coding: utf-8

# # Тема “Обучение без учителя”

# Задание 1
# Импортируйте библиотеки pandas, numpy и matplotlib.
# Загрузите "Boston House Prices dataset" из встроенных наборов 
# данных библиотеки sklearn.
# Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test)
# с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 20% от всех данных, при этом аргумент random_state должен быть равен 42.
# Масштабируйте данные с помощью StandardScaler.
# Постройте модель TSNE на тренировочный данных с параметрами:
# n_components=2, learning_rate=250, random_state=42.
# Постройте диаграмму рассеяния на этих данных.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()


# In[3]:


boston.keys()


# In[4]:


data = boston["data"]


# In[5]:


feature_names = boston["feature_names"]

feature_names


# In[6]:


target = boston["target"]

target[:10]


# In[7]:


X = pd.DataFrame(data, columns=feature_names)

X.head()


# In[9]:


y = pd.DataFrame(target, columns=["price"])
y.head()


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[15]:


X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[16]:


from sklearn.manifold import TSNE


# In[17]:


tsne = TSNE(n_components=2, learning_rate=250, random_state=42)

X_train_tsne = tsne.fit_transform(X_train_scaled)


# In[18]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])

plt.show()


# In[ ]:





# Задание 2
# С помощью KMeans разбейте данные из тренировочного набора на 3 кластера,
# используйте все признаки из датафрейма X_train.
# Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
# Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE,
# и раскрасьте точки из разных кластеров разными цветами.
# Вычислите средние значения price и CRIM в разных кластерах.
# 

# In[23]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, max_iter=100)

labels_train = kmeans.fit_predict(X_train_scaled)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)

plt.show()


# In[24]:


y_train[labels_train == 0].mean()


# In[25]:


y_train[labels_train == 1].mean()


# In[26]:


y_train[labels_train == 2].mean()


# In[28]:


X_train.CRIM[labels_train == 0].mean()


# In[29]:


X_train.CRIM[labels_train == 1].mean()


# In[30]:


X_train.CRIM[labels_train == 2].mean()


# In[ ]:




