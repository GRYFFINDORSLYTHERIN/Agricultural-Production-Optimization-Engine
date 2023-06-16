#!/usr/bin/env python
# coding: utf-8

# # Agricultural-Production-Optimization-Engine

# 
# As we all know that agriculture depends largely on the nature of soil and the climatic conditions and many a times, we face unpredictable changes in climate like, non-seasonal rainfall or heat waves or fluctuations in humidity levels, etc. and all such events cause a great loss to our farmers and farming, because of which they are not able to utilize their agricultural land to it's fullest.So to solve all such problems, I have build a Machine Learning Model by the virtue of which we can help farmers, optimize the agricultural production, because this predictive model will help them understand that for a particular soil & given climatic condition, which crop will be best suitable for the harvest.
# 
# There are 7 key factors that I've taken into account which will help us in determining, exactly which crop should be grown and at what period of time, viz. Amount of Nitrogen, Phosphorus and Potassium in soil, Temperature in degree celcius, Humidity, pH and Rainfall in mm.
# 
# Tools used: Python & Jupyter Notebook Libraries used: Numpy, Pandas, Seaborn, Matplotlib, ipywidgets and sklearn. Machine Learning Algorithms used: Clustering Analysis and Logistic Regression.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact


# In[2]:


df = pd.read_csv('data.csv')
df.head(2)


# In[3]:


print('Shape of Dataset :' , df.shape)


# In[4]:


df.isnull().sum()


# In[5]:


df.info


# In[6]:


df['label'].value_counts()


# In[7]:


#lets' check the summary for all the crops.
print('Average Ratio of Nitrogen in the Soil : {0:.2f}'.format(df['N'].mean()))
print('Average Ratio of Phosphorous in the Soil : {0:.2f}'.format(df['P'].mean()))
print('Average Ratio of Potassium in the Soil : {0:.2f}'.format(df['K'].mean()))
print('Average Temperature in Celsius in the Soil : {0:.2f}'.format(df['temperature'].mean()))
print('Average Relative Humidity in % : {0:.2f}'.format(df['humidity'].mean()))
print('Average PH value of the Soil : {0:.2f}'.format(df['ph'].mean()))
print('Average Rainfall in mm of the Soil : {0:.2f}'.format(df['rainfall'].mean()))


# In[8]:


#lets check the summary statistics for each of the crops

@interact
def summary(crops = list(df['label'].value_counts().index)):
    x =  df[df['label'] == crops]
    print('---------------------------------------------------------')
    print('Statistics for Nitrogen')
    print('Minium Nitrogen required :' , x['N'].min())
    print('Average Nitrogen required :' , x['N'].mean())
    print('Maximum Nitrogen required :' , x['N'].max())
    print('---------------------------------------------------------')
    print('Statistics for Phosphorous')
    print('Minium Phosphorous required :' , x['P'].min())
    print('Average Phosphorous required :' , x['P'].mean())
    print('Maximum Phosphorous required :' , x['P'].max())
    print('---------------------------------------------------------')
    print('Statistics for Potassium')
    print('Minium Potassium required :' , x['K'].min())
    print('Average Potassium required :' , x['K'].mean())
    print('Maximum Potassium required :' , x['K'].max())
    print('---------------------------------------------------------')
    print('Statistics for Temperature')
    print('Minium Temperature required in celsius: {0:.2f}' , x['temperature'].min())
    print('Average Temperature required in celsius : {0:.2f}' , x['temperature'].mean())
    print('Maximum Temperature required in celsius : {0:.2f}',x['temperature'].max())
    print('---------------------------------------------------------')
    print('Statistics for Relative Humidity')
    print('Minium Relative Humidity required : {0:.2f}' , x['humidity'].min())
    print('Average Relative Humidity required : {0:.2f}' , x['humidity'].mean())
    print('Maximum Relative Humidity required : {0:.2f}', x['humidity'].max())
    print('---------------------------------------------------------')
    print('Statistics for PH in mm')
    print('Minium PH required in mm : {0:.2f}' , x['ph'].min())
    print('Average PH required in mm: {0:.2f}',x['ph'].mean())
    print('Maximum PH required in mm: {0:.2f}',x['ph'].max())
    print('---------------------------------------------------------')
    print('Statistics for Rainfall')
    print('Minium Rainfall required : {0:.2f}',x['rainfall'].min())
    print('Average Rainfall required : {0:.2f}',x['rainfall'].mean())
    print('Maximum Rainfall required : {0:.2f}',x['rainfall'].max())


# In[9]:


#lets compare the average requirement for each crops with average conditions.
@interact
def compare(conditions = ['N' , 'P' , 'K' ,'temperature' , 'ph' , 'humidity' , 'rainfall']):
    print('Average value for' , conditions , 'is {0:.2f}'.format(df[conditions].mean()))
    print('-----------------------------------------------------------------------------')
    print('Rice :{0:.2f}'.format(df[(df['label'] == 'rice')][conditions].mean()))
    print('Black Grams :{0:.2f}'.format(df[(df['label'] == 'blackgram')][conditions].mean()))
    print('Maize :{0:.2f}'.format(df[(df['label'] == 'maize')][conditions].mean()))
    print('Jute :{0:.2f}'.format(df[(df['label'] == 'jute')][conditions].mean()))
    print('Coconut :{0:.2f}'.format(df[(df['label'] == 'coconut')][conditions].mean()))
    print('Apple :{0:.2f}'.format(df[(df['label'] == 'apple')][conditions].mean()))
    print('Papaya :{0:.2f}'.format(df[(df['label'] == 'papaya')][conditions].mean()))
    print('Muskmelon :{0:.2f}'.format(df[(df['label'] == 'muskmelon')][conditions].mean()))
    print('Water melon :{0:.2f}'.format(df[(df['label'] == 'watermelon')][conditions].mean()))
    print('Kidney Beans :{0:.2f}'.format(df[(df['label'] == 'kidneybeans')][conditions].mean()))
    print('Mung Beans :{0:.2f}'.format(df[(df['label'] == 'mungbean')][conditions].mean()))
    print('Oranges :{0:.2f}'.format(df[(df['label'] == 'orange')][conditions].mean()))
    print('Chick Peas :{0:.2f}'.format(df[(df['label'] == 'chickpea')][conditions].mean()))
    print('Lentils :{0:.2f}'.format(df[(df['label'] == 'lentil')][conditions].mean()))
    print('Moth Beans :{0:.2f}'.format(df[(df['label'] == 'mothbeans')][conditions].mean()))
    print('Pigeon Peas :{0:.2f}'.format(df[(df['label'] == 'pigeonpeas')][conditions].mean()))
    print('Mango :{0:.2f}'.format(df[(df['label'] == 'mango')][conditions].mean()))
    print('Pomegranate :{0:.2f}'.format(df[(df['label'] == 'pomegranate')][conditions].mean()))
    print('coffee :{0:.2f}'.format(df[(df['label'] == 'coffee')][conditions].mean()))
    print('Cotton :{0:.2f}'.format(df[(df['label'] == 'cotton')][conditions].mean()))
    print('Grapes :{0:.2f}'.format(df[(df['label'] == 'grapes')][conditions].mean()))
    print('Banana :{0:.2f}'.format(df[(df['label'] == 'banana')][conditions].mean()))
    


# In[10]:


#Lets make this functions more intuitive
@interact
def compare(conditions = ['N' , 'P' , 'K' ,'temperature' , 'ph' , 'humidity' , 'rainfall']):
    print('Crops which require greater than average' , conditions , '\n')
    print(df[df[conditions]>df[conditions].mean()]['label'].unique())
    print('-----------------------------------------------------------------')
    print('Crops which require less than average' , conditions , '\n')
    print(df[df[conditions] <= df[conditions].mean()]['label'].unique())


# In[15]:


#Checking distributiion for each crop

plt.subplot(3,3,1)
sns.histplot(df['N'], color="yellow")
plt.xlabel('Nitrogen', fontsize = 12)
plt.figure(figsize=(12,8))
plt.grid()

plt.subplot(3,3,2)
sns.histplot(df['P'], color="orange")
plt.xlabel('Phosphorous', fontsize = 12)
plt.figure(figsize=(12,12))
plt.grid()

plt.subplot(3,3,3)
sns.histplot(df['K'], color="darkblue")
plt.xlabel('Pottasium', fontsize = 12)
plt.figure(figsize=(12,12))
plt.grid()

plt.subplot(3,4,4)
sns.histplot(df['temperature'], color="black")
plt.xlabel('Temperature', fontsize = 12)
plt.figure(figsize=(12,12))
plt.grid()

plt.subplot(2,4,5)
sns.histplot(df['rainfall'], color="grey")
plt.xlabel('Rainfall', fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.histplot(df['humidity'], color="lightgreen")
plt.xlabel('Humidity', fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.histplot(df['ph'], color="darkgreen")
plt.xlabel('PH Level', fontsize = 12)
plt.grid()

plt.suptitle('Distribution for Agricultural Conditions', fontsize = 20)
plt.show()


# In[16]:


#Checking that crops those have unusual requirements

print("Some Interesting Patterns")
print("...........................................")
print("Crops that require very High Ratio of Nitrogen Content in Soil:", df[df['N'] > 120]['label'].unique())
print("Crops that require very High Ratio of Phosphorous Content in Soil:", df[df['P'] > 100]['label'].unique())
print("Crops that require very High Ratio of Potassium Content in Soil:", df[df['K'] > 200]['label'].unique())
print("Crops that require very High Rainfall:", df[df['rainfall'] > 200]['label'].unique())
print("Crops that require very Low Temperature:", df[df['temperature'] < 10]['label'].unique())
print("Crops that require very High Temperature:", df[df['temperature'] > 40]['label'].unique())
print("Crops that require very Low Humidity:", df[df['humidity'] < 20]['label'].unique())
print("Crops that require very Low pH:", df[df['ph'] < 4]['label'].unique())
print("Crops that require very High pH:", df[df['ph'] > 9]['label'].unique())


# In[17]:


#Checking which crop to be grown according to the season

print("Summer Crops")
print(df[(df['temperature'] > 30) & (df['humidity'] > 50)]['label'].unique())
print("...........................................")
print("Winter Crops")
print(df[(df['temperature'] < 20) & (df['humidity'] > 30)]['label'].unique())
print("...........................................")
print("Monsoon Crops")
print(df[(df['rainfall'] > 200) & (df['humidity'] > 30)]['label'].unique())


# In[18]:


from sklearn.cluster import KMeans

#removing the labels column
x = df.drop(['label'], axis=1)

#selecting all the values of data
x = x.values

#checking the shape
print(x.shape)


# In[34]:


#Determining the optimum number of clusters within the Dataset

plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 2000, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
#Plotting the results

plt.plot(range(1,11), wcss)
plt.title('Elbow Method', fontsize = 20)
plt.xlabel('No of Clusters')
plt.ylabel('wcss')
plt.show


# In[35]:


#Implementation of K Means algorithm to perform Clustering analysis

km = KMeans(n_clusters = 4, init = 'k-means++',  max_iter = 2000, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

#Finding the results
a = df['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

#Checking the clusters for each crop
print("Lets Check the results after applying K Means Clustering Analysis \n")
print("Crops in First Cluster:", z[z['cluster'] == 0]['label'].unique())
print("...........................................")
print("Crops in Second Cluster:", z[z['cluster'] == 1]['label'].unique())
print("...........................................")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())
print("...........................................")
print("Crops in Fourth Cluster:", z[z['cluster'] == 3]['label'].unique())


# In[36]:


#Splitting the Dataset for predictive modelling

y =df['label']
x = df.drop(['label'], axis=1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[37]:


#Creating training and testing sets for results validation
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("The Shape Of x train:", x_train.shape)
print("The Shape Of x test:", x_test.shape)
print("The Shape Of y train:", y_train.shape)
print("The Shape Of y test:", y_test.shape)


# In[38]:


#Creating a Predictive Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[39]:


#Evaluating the model performance
from sklearn.metrics import confusion_matrix

#Printing the Confusing Matrix
plt.rcParams['figure.figsize'] = (10,10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'inferno')
plt.title('Confusion Matrix For Logistic Regression', fontsize = 15)
plt.show()


# In[40]:


#Defining the classification Report
from sklearn.metrics import classification_report

#Printing the Classification Report
cr = classification_report(y_test, y_pred)
print(cr)


# In[41]:


#head of dataset
df.head()


# In[44]:


prediction = model.predict((np.array([[90, 40, 40, 20, 80, 7, 200]])))
print("The Suggested Crop for given climatic condition is :",prediction)

