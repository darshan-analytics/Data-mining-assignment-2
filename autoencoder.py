#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing
sns.set_style("whitegrid")

np.random.seed(697)
from sklearn.preprocessing import MinMaxScaler


# In[199]:


df = pd.read_csv("water-treatment.data", header = None, sep= ",")


# In[200]:


df1 = df.drop(df.columns[0], axis = 1)


# In[201]:


df2 = df1.replace(to_replace = '?', value = 'Nan')
df12 = df2.replace(to_replace = 'Nan', value = df2.median(axis = 0))


# In[202]:


df15 = df12
for i in range(1,39):
    df15[i] = pd.to_numeric(df15[i], errors='ignore')


# In[203]:


df15 = pd.DataFrame(df15)


# In[204]:


df16 = pd.DataFrame(df15)
df16 = df16.values
scaler = MinMaxScaler()
df15_norm = scaler.fit_transform(df15)
df15_norm = pd.DataFrame(df15_norm)
df15_norm = df15_norm.values


# In[205]:


train, test_df = train_test_split(df15_norm, test_size = 0.15, random_state= 42)
train_df, dev_df = train_test_split(train, test_size = 0.15, random_state= 42)


# In[206]:


train_df.sum()/train_df.shape[0] #0.2210
dev_df.sum()/dev_df.shape[0] #0.2269
test_df.sum()/test_df.shape[0] #0.2168


# In[207]:


train_y = train_df
dev_y = dev_df
test_y = test_df


# In[208]:


train_x = train_df
dev_x = dev_df
test_x = test_df


# In[209]:


train_x =np.array(train_x)
dev_x =np.array(dev_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
dev_y = np.array(dev_y)
test_y = np.array(test_y)


# In[210]:


encoding_dim = 8 #as the PCA


# In[211]:


input_data = Input(shape=(train_x.shape[1],))


# In[212]:


encoded = Dense(encoding_dim, activation='elu')(input_data)


# In[213]:


decoded = Dense(train_x.shape[1], activation='sigmoid')(encoded)


# In[214]:


autoencoder = Model(input_data, decoded)


# In[215]:


autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy')


# In[216]:


hist_auto = autoencoder.fit(train_x, train_x,
                epochs=50,
                batch_size=30,
                shuffle=True,
                validation_data=(dev_x, dev_x))


# In[217]:


plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[218]:


encoder = Model(input_data, encoded)


# In[219]:


encoded_input = Input(shape=(encoding_dim,))


# In[220]:


decoder_layer = autoencoder.layers[-1]


# In[221]:


decoder = Model(encoded_input, decoder_layer(encoded_input))


# In[222]:


encoded_x = encoder.predict(test_x)
decoded_output = decoder.predict(encoded_x)


# In[223]:


encoded_train_x = encoder.predict(train_x)
encoded_test_x = encoder.predict(test_x)


# In[224]:


model = Sequential()
model.add(Dense(8, input_dim=encoded_train_x.shape[1],
                kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"
                )
          )


# In[225]:


model.add(Dropout(0.2))


# In[226]:


model.add(Dense(38))


# In[227]:


model.add(Activation("sigmoid"))


# In[228]:


model.compile(loss="binary_crossentropy", optimizer='adam')


# In[229]:


history = model.fit(encoded_train_x, train_y, validation_split=0.2, epochs=10, batch_size=64)


# In[230]:


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Encoded model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[231]:


predictions_NN_prob = model.predict(encoded_test_x)
predictions_NN_prob = predictions_NN_prob[:,0]
predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0)


# In[232]:


predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0)

