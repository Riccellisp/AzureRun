#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 01:09:56 2022

@author: riccelli
"""
import tensorflow as tf
from sklearn.metrics import accuracy_score


import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
# Categories=['Cars','Ice cream cone','Cricket ball']
flat_data_arr=[] #input array
data = []
target_arr=[] #output array
datadir='ScalogramFigures/0' 
#path which contains all the categories of images
# for i in Categories:
    
# path=os.path.join(datadir,i)
for img in os.listdir(datadir):
    img_array=imread(os.path.join(datadir,img))
    # breakpoint()
    img_resized=resize(img_array,(100,100,4))
    data.append(img_resized)
    #flat_data_arr.append(img_resized.flatten())
    
    
data=np.array(data)

data = data.reshape(data.shape[0],100,100,4)


df1 = pd.read_csv('dataset_independent_channels.csv')
target = df1['label']
target=np.array(target)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.20,random_state=77,stratify=target)

# Our vectorized labels
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(100,100,4))
)
# [9,9] -> Conv(3,3) -> slide along the image. Stride defines this.
model.add(
    tf.keras.layers.MaxPool2D(pool_size=(1,1))
)
model.add(
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')
)
model.add(
    tf.keras.layers.MaxPool2D(pool_size=(1,1))
)

# -------------------- Classifier -----------------------

# Flatten() -> hidden layer with 128 neurons -> output layer with 10 neurons(softmax)

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
# n_classes = 10. Digitis from 0 to 9
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=128, # [X1, X2, ... X128] -> Update, [X128, X129, ... X256] -> Update, ...
    epochs=10,
    validation_data=(
        x_test, y_test
    )
)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()