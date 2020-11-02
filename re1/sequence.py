import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
import numpy as np


train_data = datasets.load_iris().data
train_label = datasets.load_iris().target

print(train_data.shape)
print(train_label.shape)

model = tf.keras.models.Sequential([
    keras.layers.Dense(3,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_data,train_label,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)

model.summary()

