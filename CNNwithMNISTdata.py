#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import random
import numpy as np

random.seed(42)
np.random.seed(42)
#tf.set_random_seed(42)


# In[8]:


cnnModel=models.Sequential()

cnnModel.add((layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1))))
cnnModel.add(layers.MaxPooling2D((2,2)))

cnnModel.add((layers.Conv2D(64,(3,3),activation="relu")))
cnnModel.add(layers.MaxPooling2D((2,2)))

cnnModel.add((layers.Conv2D(64,(3,3),activation="relu")))
cnnModel.summary()


# In[9]:


cnnModel.add(layers.Flatten())
cnnModel.add(layers.Dense(64,activation="relu"))
cnnModel.add(layers.Dense(32,activation="relu"))
cnnModel.add(layers.Dense(10,activation="softmax"))
cnnModel.summary()


# In[11]:


mnist=tf.keras.datasets.mnist

(X_train,y_train),(X_test,y_test)=mnist.load_data()

X_train=X_train.reshape((60000,28,28,1))
X_train=X_train.astype('float32')/255

X_test=X_test.reshape((10000,28,28,1))
X_test=X_test.astype('float32')/255

y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)

cnnModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

tbCallBack=tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True,write_images=True)
cnnModel.fit(X_train,y_train,epochs=5)


# In[16]:


testloss, testAccuracy=cnnModel.evaluate(X_test,y_test)
print(testAccuracy)


# In[ ]:




