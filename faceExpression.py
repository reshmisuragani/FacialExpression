#!/usr/bin/env python
# coding: utf-8

# In[8]:


from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
#from scikit_learn import train_test_split
from sklearn.model_selection import train_test_split
model = Sequential()


# In[9]:


import pandas as pd
import cv2
import numpy as np
image_size=(48,48)

def load_fer2013(dataset_path):
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

faces, emotions = load_fer2013('fer2013.csv')

xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)


# In[10]:



model = Sequential()

model.add(Conv2D(64, 3, input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.6))

model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[16]:


print(xtrain.shape)


# In[6]:


batch_size = 64
history = model.fit(np.array(xtrain), np.array(ytrain),batch_size=batch_size,epochs= 10 ,verbose=1,
                    validation_data=(np.array(xtest), np.array(ytest)),shuffle=True)


# In[18]:


batch_size = 64
history = model.fit(np.array(xtrain), np.array(ytrain),batch_size=batch_size,epochs= 15 ,verbose=1,
                    validation_data=(np.array(xtest), np.array(ytest)),shuffle=True)


# In[7]:


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('facemodel.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


# In[8]:


batch_size = 64
history = model.fit(np.array(xtrain), np.array(ytrain),batch_size=batch_size,epochs= 100 ,verbose=1,
                    validation_data=(np.array(xtest), np.array(ytest)),shuffle=True)



# In[11]:


train_acc = model.evaluate(xtrain, ytrain, verbose=0)
test_acc = model.evaluate(xtest, ytest, verbose=0)
print("Training_Accuracy: %.2f%%" % (train_acc[1]*100))
print("Test_Accuracy: %.2f%%" % (test_acc[1]*100))


# In[12]:


from keras.models import load_model

model.save('expression.model')


# In[12]:


import matplotlib.pyplot as plt
fig1 = plt.figure()
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves :CNN',fontsize=16)
plt.show()
#plt.savefig('loss.png')


# In[21]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['accuracy', 'Validation accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('accuracy',fontsize=16)
plt.title('accuracy :CNN',fontsize=16)
plt.show()
plt.savefig('accuracy.png')








