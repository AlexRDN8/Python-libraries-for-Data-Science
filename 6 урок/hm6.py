#!/usr/bin/env python
# coding: utf-8

# # Тема “Обучение с учителем”

# Задание 1
# Импортируйте библиотеки pandas и numpy.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 30% от всех данных, при этом аргумент random_state должен быть равен 42.
# Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.
# Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
# Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.
# 

# In[36]:


import numpy as np
import pandas as pd

from sklearn.datasets import load_boston


# In[37]:


boston = load_boston()

boston.keys()


# In[38]:


data = boston["data"]

data.shape


# In[39]:


feature_names = boston["feature_names"]

feature_names


# In[40]:


print(boston["DESCR"])


# In[41]:


target = boston["target"]

target[:10]


# In[42]:


X = pd.DataFrame(data, columns=feature_names)

X.head()


# In[43]:


y = pd.DataFrame(target, columns=["price"])

y.info()


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[46]:


from sklearn.linear_model import LinearRegression


# In[47]:


lr = LinearRegression()


# In[48]:


lr.fit(X_train, y_train)


# In[49]:


y_pred = lr.predict(X_test)

y_pred.shape


# In[51]:


check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})

check_test.head(10)


# In[52]:


from sklearn.metrics import r2_score


# In[53]:


r2_score(y_test, y_pred)


# In[ ]:





# Задание 2 Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble. Сделайте агрумент n_estimators равным 1000, max_depth должен быть равен 12 и random_state сделайте равным 42. Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression, но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0], чтобы получить из датафрейма одномерный массив Numpy, так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо датафрейма. Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания. Напишите в комментариях к коду, какая модель в данном случае работает лучше.
# 
# ​

# In[54]:


from sklearn.ensemble import RandomForestRegressor


# In[55]:


model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(X_train, y_train.values[:, 0])
y_pred = model.predict(X_test)


# In[73]:


y_pred = model.predict(X_test)

y_pred.shape


# In[74]:


r2_score(y_test, y_pred)


# Данная можель работает лучше, так как r2 более близко к 1, чем в моделе из 1 задачи (0.87>0.71)

# In[ ]:




