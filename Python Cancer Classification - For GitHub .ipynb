#!/usr/bin/env python
# coding: utf-8
##Classification With Python 2022 


# In[35]:


import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 


# In[36]:


df = pd.read_csv('../DATA/cancer_classification.csv')


# In[37]:


df.info()#serching for nulls


# In[38]:


df.describe()#search for features 


# In[40]:


sns.countplot(x='benign_0__mal_1',data=df)


# In[41]:


#check the corrlation between label and values 
df.corr()['benign_0__mal_1'].sort_values()


# In[42]:


#we can plot it and drop our label 
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# In[43]:


#check the corrolations between features to themselfs with heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())


# In[44]:


#TRANING THE MODEL
X = df.drop('benign_0__mal_1',axis=1)
y = df['benign_0__mal_1']


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[47]:


#scaling the data
from sklearn.preprocessing import MinMaxScaler


# In[48]:


scaler = MinMaxScaler()


# In[49]:


X_train = scaler.fit_transform(X_train)


# In[50]:


X_test = scaler.fit_transform(X_test)


# PART 2 creating the model preventing over fitting how to make sure not to over run our data set

# In[51]:


from tensorflow.keras.models import Sequential


# In[52]:


from tensorflow.keras.layers import Dense


# In[53]:


X_train.shape


# In[60]:


model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))

#BINARY CLASSIFICATION 1 OR 0 well use sigmoid function! on output 1 neuron

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# In[61]:


model.fit(x=X_train, y=y_train,epochs=600,validation_data=(X_test,y_test))


# In[68]:


#plot out the loss! the traning and validation 
losses = pd.DataFrame(model.history.history)
losses.plot()


# In[71]:


# we can see clearly were overfitting !
# because our traning loss is still going down!
#our validation data is going to increase (ORANGE) and getting worse and worse and
#needs to be stopping! at training before it gets out of hand!!#We trained to much! 


# #Will be recrating the model!

# In[73]:


model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))

#BINARY CLASSIFICATION 1 OR 0 well use sigmoid function! on output 1 neuron

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# In[74]:


from tensorflow.keras.callbacks import EarlyStopping


# In[75]:


help(EarlyStopping)


# In[78]:


# we will be monitoring our validation loss (the orange line )
# we can specify the minimum change required ,also there num of epochs with no improvement 
# to be stopped!
early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=25,verbose=1)
#we chose mode min to minimize loss,paticence to wait 25 ephocs even if error detected
#because of noise coud accur, verbose is 1 to report back


# In[79]:


model.fit(x=X_train, y=y_train,epochs=600,validation_data=(X_test,y_test),
          callbacks=[early_stop])


# In[80]:


#As i can see my fitting stopped at 65 ! so the early stop chose the perfect epoch for me.


# In[83]:


#check losses 
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[84]:


#MUCH MUCH BETTER ! its flatting out!(Oragnge)


# #3rd way adding a dropout layer 
# #drop out layers will turn off precntile of neurons randomly

# In[85]:


from tensorflow.keras.layers import Dropout


# In[118]:


model = Sequential()

model.add(Dense(30,activation='relu'))
#add dropout call
model.add(Dropout(rate=0.5))
#rate is the probability im going to turn off
#the actual neurons
#1 is 100% of neurons are going to be off 0 is none 
#there are common numbers 0.5-0.2 so 20-50% are going to be turned off 
#randomly. its not the same every time the randomly change !
model.add(Dense(15,activation='relu'))
#add dropout call
model.add(Dropout(rate=0.5))

#BINARY CLASSIFICATION 1 OR 0 well use sigmoid function! on output 1 neuron

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# In[119]:


#fit the model
model.fit(x=X_train, y=y_train,epochs=600,validation_data=(X_test,y_test),
          callbacks=[early_stop])


# In[120]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[121]:


#this is amazing resault! dropout layer + Early stopping!


# #FULL EVALUATION ON OUR CLASSES 

# In[122]:


#predicting a 0 or 1 
#in kares we predict_classes but its not exist so i used another func who did the same results
#for classification
predictions = (model.predict(X_test) > 0.5).astype("int32")


# In[123]:


from sklearn.metrics import classification_report,confusion_matrix


# In[127]:


print(classification_report(y_test,predictions))


# In[128]:


print(confusion_matrix(y_test,predictions))


