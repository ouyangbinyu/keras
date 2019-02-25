import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
print(tf.VERSION)
print(tf.keras.__version__)


model = tf.keras.Sequential()
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))

model.compile(optimizer = tf.train.AdamOptimizer(0.01),loss = tf.keras.losses.categorical_crossentropy, metrics = ['accuracy'])



data = np.random.random((1000,32))
labels = np.random.random((1000,10))

model.fit(data,labels,epochs=10,batch_size=32)